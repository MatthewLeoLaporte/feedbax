from math import sqrt
from typing import Literal, Optional, Type, TypeVar

import equinox as eqx
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from jaxtyping import Array, Float, PyTree
from plotly.colors import convert_colors_to_same_type

from feedbax import tree_labels
from feedbax._tree import tree_prefix_expand, tree_zip
from feedbax.plot.colors import (
    DEFAULT_COLORS,
    color_add_alpha,
)
from feedbax.plot.misc import AggMode, _agg_circular, _agg_standard, _maybe_unwrap
from jax_cookbook import MaskedArray

T = TypeVar("T")


def _unwrap_masked_array(x):
    """Convert MaskedArray to regular array with masked values set to NaN."""
    if isinstance(x, MaskedArray):
        return np.asarray(x.unwrap())
    return x


def profiles(
    vars_: PyTree[Float[Array, "*batch timestep"], " T"],
    keep_axis: Optional[PyTree[int, " T ..."]] = None,
    mode: Literal["std", "curves"] = "std",
    stride_curves: int = 1,
    timesteps: Optional[PyTree[Float[Array, " timestep"], " T"]] = None,
    varname: str = "Value",
    legend_title: Optional[str] = None,
    labels: Optional[PyTree[str, "T"]] = None,
    colors: Optional[list[str]] = None,
    error_bars_alpha: float = 0.2,
    n_std_plot: int = 1,
    hline: Optional[dict] = None,
    layout_kws: Optional[dict] = None,
    scatter_kws: Optional[dict] = None,
    curves_kws: Optional[dict] = None,
    agg_mode: AggMode = "standard",
    fig: Optional[go.Figure] = None,
) -> go.Figure:
    """Plot 1D state profiles as lines with error bands.

    `keep_axis` will retain one dimension of data, and plot one mean+/-error curve for
    each entry in that axis.

    `agg_mode` selects aggregation per node: "standard" uses linear mean±std,
    "circular" uses circular mean and circular std-derived bands.

    Supports MaskedArray leaves (duck-typed via .data/.mask attributes), which are
    converted to NaN-masked arrays for aggregation.
    """
    from jax_cookbook import MaskedArray, is_type

    # Treat MaskedArray as a leaf throughout processing
    is_leaf_ma = is_type(MaskedArray)

    if fig is None:
        fig = go.Figure()

    if scatter_kws is None:
        scatter_kws = dict()

    if curves_kws is None:
        curves_kws = dict()

    if timesteps is None:
        timesteps = jt.map(
            lambda x: np.arange(x.data.shape[-1] if isinstance(x, MaskedArray) else x.shape[-1]),
            vars_,
            is_leaf=is_leaf_ma,
        )
    else:
        timesteps = tree_prefix_expand(timesteps, vars_, is_leaf=is_leaf_ma)

    if labels is None:
        labels = tree_labels(vars_, is_leaf=is_leaf_ma)

    batch_axes = jt.map(
        lambda x: tuple(range((x.data if isinstance(x, MaskedArray) else x).ndim - 1)),
        vars_,
        is_leaf=is_leaf_ma,
    )
    if keep_axis is None:
        mean_axes = batch_axes
    else:
        mean_axes = jt.map(
            lambda axes, axis: tuple(ax for ax in axes if ax != axis),
            batch_axes,
            tree_prefix_expand(keep_axis, vars_, is_leaf=is_leaf_ma),
            is_leaf=lambda x: isinstance(x, tuple) and eqx.is_array_like(x[0]),
        )

    # Unwrap MaskedArray for aggregation
    vars_unwrapped = jt.map(_unwrap_masked_array, vars_, is_leaf=is_leaf_ma)

    agg_fn = _agg_standard if agg_mode == "standard" else _agg_circular
    # Aggregate per node (return mean, upper, lower)
    means, ubs, lbs = jtree.unzip(
        jt.map(
            lambda x, axis: agg_fn(x, axis, n_std_plot),
            vars_unwrapped,
            mean_axes,
            is_leaf=is_leaf_ma,
        )
    )

    means, ubs, lbs = jtree.unzip(
        jt.map(lambda m, ub, lb: _maybe_unwrap(m, ub, lb, agg_mode), means, ubs, lbs, is_leaf=is_leaf_ma)
    )

    if keep_axis is None:
        means, ubs, lbs = jt.map(lambda arr: arr[None, ...], (means, ubs, lbs), is_leaf=is_leaf_ma)
        vars_flat = jt.map(lambda x: np.reshape(x, (1, -1, x.shape[-1])), vars_unwrapped, is_leaf=is_leaf_ma)
    else:
        vars_flat = jt.map(
            lambda x: np.reshape(
                np.moveaxis(x, keep_axis, 0),
                (x.shape[keep_axis], -1, x.shape[-1]),
            ),
            vars_unwrapped,
            is_leaf=is_leaf_ma,
        )

    if colors is None:
        colors = DEFAULT_COLORS

    colors_rgb: list[str]
    colors_rgb, _ = convert_colors_to_same_type(colors, colortype="rgb")  # type: ignore

    if hline is not None:
        fig.add_hline(**hline)

    if agg_mode == "circular":
        fig.add_hline(y=np.pi, line_dash="dot", line_color="grey")
        fig.add_hline(y=-np.pi, line_dash="dot", line_color="grey")

    def add_profile(fig, label, var_flat, means, ubs, lbs, ts, color) -> go.Figure:
        traces = []

        for i, (curves, mean, ub, lb) in enumerate(zip(var_flat, means, ubs, lbs)):
            traces.append(
                # Mean
                go.Scatter(
                    name=label,
                    showlegend=(i == 0 and label is not None),
                    legendgroup=label,
                    x=ts,
                    y=mean,
                    marker_size=3,
                    line=dict(color=color),
                    **scatter_kws,
                )
            )

            if mode == "curves":
                # TODO: show exact trial indices in hoverinfo
                for curve in curves[::stride_curves]:
                    traces.append(
                        go.Scatter(
                            name=label,
                            showlegend=False,
                            legendgroup=label,
                            x=ts,
                            y=curve,
                            mode="lines",
                            line=dict(color=color, width=0.5),
                            **curves_kws,
                        )
                    )

            elif mode == "std":
                traces.extend(
                    [  # Bounds
                        go.Scatter(
                            name="Upper bound",
                            legendgroup=label,
                            x=ts,
                            y=ub,
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        go.Scatter(
                            name="Lower bound",
                            legendgroup=label,
                            x=ts,
                            y=lb,
                            line=dict(color="rgba(255,255,255,0)"),
                            fill="tonexty",
                            fillcolor=color_add_alpha(
                                color, error_bars_alpha / sqrt(means.shape[0])
                            ),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                    ]
                )

            else:
                raise ValueError(f"Invalid mode: {mode}")

        fig.add_traces(traces)
        return fig

    # Treat MaskedArray as a leaf and tuples as leaves when flattening for plotting
    plot_data = jt.leaves(
        tree_zip(vars_flat, means, ubs, lbs, timesteps, labels, is_leaf=is_leaf_ma),
        is_leaf=lambda x: isinstance(x, tuple) or isinstance(x, MaskedArray),
    )

    for i, (var_flat, means_i, ubs_i, lbs_i, ts, label) in enumerate(plot_data):
        fig = add_profile(
            fig,
            label,
            var_flat,
            means_i,
            ubs_i,
            lbs_i,
            ts,
            colors_rgb[i],
        )

    fig.update_layout(
        width=800,
        height=500,
        xaxis_title="Time step",
        yaxis_title=varname,
        # yaxis_tickformat="e",
        yaxis_exponentformat="E",
        margin=dict(l=80, r=10, t=30, b=60),
        legend_traceorder="reversed",
    )

    fig.update_layout(legend_itemsizing="constant")
    fig.update_layout(legend_title_text=("Condition" if legend_title is None else legend_title))

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


def profile(
    var: Float[Array, "batch timestep"],
    var_label: str = "Value",
    colors: Optional[list[str]] = None,
    layout_kws: Optional[dict] = None,
    **kwargs,
) -> go.Figure:
    """Plot a single batch of lines."""
    # TODO: vlines
    fig = px.line(
        var.T,
        color_discrete_sequence=colors,
        labels=dict(index="Time step", value=var_label, variable="Trial"),
        **kwargs,
    )
    if layout_kws is not None:
        fig.update_layout(layout_kws)
    return fig
