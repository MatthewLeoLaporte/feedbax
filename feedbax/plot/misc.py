from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, NamedTuple, Optional, TypeAlias, TypeGuard, TypeVar

import equinox as eqx
import jax.tree as jt
import numpy as np
import plotly.graph_objs as go
import polars as pl
from jaxtyping import Array, PyTree, Shaped
from plotly.basedatatypes import BaseTraceType

from feedbax import tree_labels

AggMode: TypeAlias = Literal["standard", "circular"]

StrMapping: TypeAlias = Mapping[str, Any]

T = TypeVar("T")


def _wrap_to_pi(x):
    TAU = 2 * np.pi
    return (x + np.pi) % TAU - np.pi


def _unwrap_time(x, axis: int = -1):
    """Unwrap a phase/angle sequence along `axis` to remove ±2π jumps."""
    dx = np.diff(x, axis=axis)
    dx_mod = (dx + np.pi) % (2 * np.pi) - np.pi  # bring diffs to (-π, π]
    shifts = dx_mod - dx  # multiples of 2π to add
    corr = np.cumsum(shifts, axis=axis)
    pad_shape = list(x.shape)
    pad_shape[axis] = 1
    corr = np.concatenate([np.zeros(pad_shape, dtype=x.dtype), corr], axis=axis)
    return x + corr


def _unwrap_around(ref, x):
    """Shift `x` by multiples of 2π so it stays near `ref`."""
    return ref + ((x - ref + np.pi) % (2 * np.pi) - np.pi)


def _maybe_unwrap(mean, ub, lb, mode):
    if mode != "circular":
        return mean, ub, lb
    mean_u = _unwrap_time(mean, axis=-1)
    ub_u = _unwrap_around(mean_u, _unwrap_time(ub, axis=-1))
    lb_u = _unwrap_around(mean_u, _unwrap_time(lb, axis=-1))
    return mean_u, ub_u, lb_u


def _agg_standard(x, axis, n_std_plot: int):
    """Standard (linear) aggregation: (mean, upper, lower)."""
    mean = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)
    ub = mean + n_std_plot * std
    lb = mean - n_std_plot * std
    return mean, ub, lb


def _agg_circular(x, axis, n_std_plot: int):
    """Circular aggregation on angles in radians: (mean, upper, lower).

    Mean is the vector (circular) mean; 'std' is derived from mean resultant length:
        sigma = sqrt(-2 ln R)
    Upper/lower are mean ± n*sigma, wrapped back to (-π, π].
    """
    # mask NaNs
    mask = np.isfinite(x)
    # sums of sin/cos with NaNs suppressed
    S = np.sum(np.where(mask, np.sin(x), 0.0), axis=axis)
    C = np.sum(np.where(mask, np.cos(x), 0.0), axis=axis)
    n = np.sum(mask, axis=axis)

    mean = np.arctan2(S, C)

    # mean resultant length (avoid divide-by-zero / log(0))
    R = np.sqrt(S**2 + C**2) / np.maximum(n, 1e-12)
    R = np.clip(R, 1e-12, 1.0)

    sigma = np.sqrt(-2.0 * np.log(R))
    ub = _wrap_to_pi(mean + n_std_plot * sigma)
    lb = _wrap_to_pi(mean - n_std_plot * sigma)
    return mean, ub, lb


def arr_to_nested_tuples(arr):
    """Like `ndarray.tolist()` but ensures the bottom-most level is tuples."""
    if arr.ndim == 1:
        return tuple(arr.tolist())
    elif arr.ndim > 1:
        return [arr_to_nested_tuples(sub_arr) for sub_arr in arr]
    else:
        return arr.item()  # For 0-dimensional arrays


def is_trace(element):
    return isinstance(element, BaseTraceType)


def tree_of_2d_timeseries_to_df(
    tree: PyTree[Shaped[Array, "condition timestep xy=2"], " T"],
    labels: Optional[PyTree[str, " T"]] = None,
) -> pl.DataFrame:
    """Construct a single dataframe from a PyTree of spatial timeseries arrays,
    batched over trials."""

    array_spec = jt.map(eqx.is_array, tree)
    arrays_flat = map(np.array, jt.leaves(eqx.filter(tree, array_spec)))

    if labels is None:
        labels = tree_labels(tree, join_with=" ")

    labels_flat = jt.leaves(eqx.filter(labels, array_spec))

    def get_xy_df(array):
        # Concatenate all trials.
        return pl.concat(
            [
                # For each trial, construct all timesteps of x, y data.
                pl.from_numpy(x, schema=["x", "y"]).hstack(
                    pl.DataFrame(
                        {
                            "Timestep": np.arange(x.shape[0]),
                            "Condition": pl.repeat(i, x.shape[0], eager=True),
                        }
                    )
                )
                for i, x in enumerate(array)
            ]
        )

    xy_dfs = [get_xy_df(x) for x in arrays_flat]

    # Concatenate all variables.
    df = pl.concat(
        [
            xy_df.hstack([pl.repeat(label, len(xy_df), eager=True).alias("var")])
            for xy_df, label in zip(xy_dfs, labels_flat)
        ]
    )

    return df


def unshare_axes(fig: go.Figure):
    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)
    fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))


def _normalize_vars(vars_):
    """Ensure each leaf has shape (*batch, time, d) and d in {2,3}.
    Do **not** require batch/time shapes to match across leaves; we handle per-leaf.
    Uses jax.tree.map so dicts yield values, not keys.
    """

    def _norm(x):
        if not eqx.is_array(x):
            return x
        if x.ndim < 2:
            raise ValueError("Each array must have at least (time, dim) axes")
        if x.ndim == 2:  # (time, d) -> add singleton batch
            x = x[None, ...]
        d = x.shape[-1]
        if d not in (2, 3):
            raise ValueError(f"Final axis (dim) must be 2 or 3; got {d}")
        return x

    return jt.map(_norm, vars_)


class AxesLabels(NamedTuple):
    x: PyTree[str]
    y: PyTree[str]
    z: PyTree[str] = None


def _mean_over_axes(x: np.ndarray, axis0: int, mean_exclude_axes: Sequence[int]):
    """Mean over all batch axes except the colorscale axis and those explicitly excluded.
    Returns array shaped (C, M, time, d) where M comes from preserved axes (or 1)."""
    full_ndim = x.ndim
    batch_nd = full_ndim - 2
    exclude = set(ax if ax >= 0 else full_ndim + ax for ax in mean_exclude_axes)

    axes_to_mean = [ax for ax in range(batch_nd) if ax != axis0 and ax not in exclude]
    if axes_to_mean:
        m = np.nanmean(x, axis=tuple(axes_to_mean), keepdims=False)
    else:
        m = x

    adjusted_axis0 = axis0
    for ax in sorted(axes_to_mean):
        if ax < axis0:
            adjusted_axis0 -= 1

    m = np.moveaxis(m, adjusted_axis0, 0)  # (C, ..., time, d)
    if m.ndim == 3:  # (C, time, d)
        m = m[:, None, ...]  # (C,1,time,d)
    else:
        lead = m.shape[1:-2]
        M = int(np.prod(lead)) if lead else 1
        m = np.reshape(m, (m.shape[0], M, *m.shape[-2:]))
    return m


def _is_str_mapping(x: object) -> TypeGuard[StrMapping]:
    """True if x is a Mapping with str keys (safe for **kwargs)."""
    if not isinstance(x, Mapping):
        return False
    # Empty dict is fine; otherwise ensure keys are str
    for k in x.keys():
        if not isinstance(k, str):
            return False
    return True


def _is_mapping_of_type(
    x: object,
    is_type_: Callable[[Any], bool],
) -> TypeGuard[Mapping]:
    """True if x is a Mapping whose values are Mapping[str, Any]."""
    if not isinstance(x, Mapping):
        return False
    for v in x.values():
        if not is_type_(v):
            return False
    return True


def columns_mean_std(dfs: PyTree[pl.DataFrame], index_col: Optional[str] = None):
    if index_col is not None:
        spec = {
            index_col: pl.col(index_col),
            "mean": pl.concat_list(pl.col("*").exclude(index_col)).list.mean(),
            "std": pl.concat_list(pl.col("*").exclude(index_col)).list.std(),
        }
    else:
        spec = {
            "mean": pl.concat_list(pl.col("*")).list.mean(),
            "std": pl.concat_list(pl.col("*")).list.std(),
        }

    return jt.map(
        lambda df: df.select(**spec),
        dfs,
    )


def errorbars(col_means_stds: PyTree[pl.DataFrame], n_std: int):
    return jt.map(
        lambda df: df.select(
            lb=pl.col("mean") - n_std * pl.col("std"),
            ub=pl.col("mean") + n_std * pl.col("std"),
        ),
        col_means_stds,
    )
