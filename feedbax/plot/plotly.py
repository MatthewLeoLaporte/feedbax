"""

:copyright: Copyright 2024 by Matt L. Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from math import sqrt
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Optional,
    TypeAlias,
    TypeGuard,
    TypeVar,
    ValuesView,
)

import equinox as eqx
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
import plotly.colors as plc
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import polars as pl
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Shaped

# pyright: reportMissingTypeStubs=false
from plotly.basedatatypes import BaseTraceType
from plotly.colors import convert_colors_to_same_type, sample_colorscale
from plotly.subplots import make_subplots

from feedbax import tree_labels
from feedbax._tree import tree_prefix_expand, tree_zip
from feedbax.bodies import SimpleFeedbackState
from feedbax.loss import LossDict
from feedbax.misc import where_func_to_attr_str_tree
from feedbax.types import SeqOf, SeqOfT

if TYPE_CHECKING:
    from feedbax.task import TaskTrialSpec


logger = logging.getLogger(__name__)


T = TypeVar("T")


pio.templates.default = "plotly_white"
DEFAULT_COLORS = pio.templates[pio.templates.default].layout.colorway  # pyright: ignore


class AxesLabels(NamedTuple):
    x: PyTree[str]
    y: PyTree[str]
    z: PyTree[str] = None


def color_add_alpha(rgb_str: str, alpha: float):
    return f"rgba{rgb_str[3:-1]}, {alpha})"


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


# --- helpers ---


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


AggMode: TypeAlias = Literal["standard", "circular"]


def profiles(
    vars_: PyTree[Float[Array, "*batch timestep"], "T"],
    keep_axis: Optional[PyTree[int, "T ..."]] = None,
    mode: Literal["std", "curves"] = "std",
    stride_curves: int = 1,
    timesteps: Optional[PyTree[Float[Array, " timestep"], "T"]] = None,
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
    """
    if fig is None:
        fig = go.Figure()

    if scatter_kws is None:
        scatter_kws = dict()

    if timesteps is None:
        timesteps = jt.map(lambda x: np.arange(x.shape[-1]), vars_)
    else:
        timesteps = tree_prefix_expand(timesteps, vars_)

    if labels is None:
        labels = tree_labels(vars_)

    batch_axes = jt.map(
        lambda x: tuple(range(x.ndim - 1)),
        vars_,
    )
    if keep_axis is None:
        mean_axes = batch_axes
    else:
        mean_axes = jt.map(
            lambda axes, axis: tuple(ax for ax in axes if ax != axis),
            batch_axes,
            tree_prefix_expand(keep_axis, vars_),
            is_leaf=lambda x: isinstance(x, tuple) and eqx.is_array_like(x[0]),
        )

    agg_fn = _agg_standard if agg_mode == "standard" else _agg_circular
    # Aggregate per node (return mean, upper, lower)
    means, ubs, lbs = jtree.unzip(
        jt.map(
            lambda x, axis: agg_fn(x, axis, n_std_plot),
            vars_,
            mean_axes,
        )
    )

    means, ubs, lbs = jtree.unzip(
        jt.map(lambda m, ub, lb: _maybe_unwrap(m, ub, lb, agg_mode), means, ubs, lbs)
    )

    if keep_axis is None:
        means, ubs, lbs = jt.map(lambda arr: arr[None, ...], (means, ubs, lbs))
        vars_flat = jt.map(lambda x: np.reshape(x, (1, -1, x.shape[-1])), vars_)
    else:
        vars_flat = jt.map(
            lambda x: np.reshape(
                np.moveaxis(x, keep_axis, 0),
                (x.shape[keep_axis], -1, x.shape[-1]),
            ),
            vars_,
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
                    # legendgroup=label,
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
                            # legendgroup=label,
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
                            # legendgroup=label,
                            x=ts,
                            y=ub,
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        go.Scatter(
                            name="Lower bound",
                            # legendgroup=label,
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

    plot_data = jt.leaves(
        tree_zip(vars_flat, means, ubs, lbs, timesteps, labels),
        is_leaf=lambda x: isinstance(x, tuple),
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


def loss_history(
    losses: LossDict,
    loss_context: Literal["training", "validation"] = "training",
    colors: Optional[list[str]] = None,
    error_bars_alpha: float = 0.3,
    n_std_plot: int = 1,
    layout_kws: Optional[dict] = None,
    scatter_kws: Optional[dict] = None,
    hover_compare: bool = True,
    **kwargs,
) -> go.Figure:
    if scatter_kws is not None:
        scatter_kws = dict(hovertemplate="%{y:.2e}") | scatter_kws
    else:
        scatter_kws = dict(hovertemplate="%{y:.2e}")

    fig = go.Figure(
        layout_modebar_add=["v1hovermode"],
        **kwargs,
    )

    if loss_context == "training":
        scatter_mode = "markers+lines"
    elif loss_context == "validation":
        # Validation losses are usually sparse in time, so don't connect with lines
        scatter_mode = "markers"
    else:
        raise ValueError(f"{loss_context} is not a valid loss context")

    losses: LossDict | Array = jt.map(
        lambda x: np.array(x),
        losses,
    )

    losses_total = losses if isinstance(losses, Array) else losses.total

    timesteps = pl.DataFrame({"timestep": range(losses_total.shape[0])})

    dfs = jt.map(
        lambda losses: pl.DataFrame(losses),
        OrderedDict({"Total": losses_total}) | dict(losses),
    )

    # TODO: Only apply this when yaxis is log scaled
    dfs = jt.map(
        lambda df: df.select(
            [
                np.log10(pl.all()),
            ]
        ),
        dfs,
    )

    loss_statistics = columns_mean_std(dfs)
    error_bars_bounds = errorbars(loss_statistics, n_std_plot)

    # TODO: Only apply this when yaxis is log scaled
    loss_statistics, error_bars_bounds = jt.map(
        lambda df: timesteps.hstack(
            df.select(
                [
                    np.power(10, pl.col("*")),  # type: ignore
                ]
            )
        ),
        (loss_statistics, error_bars_bounds),
    )

    if colors is None:
        colors = DEFAULT_COLORS

    colors_rgb: list[str]
    colors_rgb, _ = convert_colors_to_same_type(colors, colortype="rgb")  # type: ignore

    colors_dict = {
        label: "rgb(0,0,0)" if label == "Total" else color for label, color in zip(dfs, colors_rgb)
    }

    for i, label in enumerate(reversed(dfs)):
        # Mean
        trace = go.Scatter(
            name=label,
            legendgroup=str(i),
            x=loss_statistics[label]["timestep"],
            y=loss_statistics[label]["mean"],
            mode=scatter_mode,
            marker_size=3,
            line=dict(color=colors_dict[label]),
        )
        trace.update(scatter_kws)
        fig.add_trace(trace)

        # Error bars
        fig.add_trace(
            go.Scatter(
                name="Upper bound",
                legendgroup=str(i),
                x=loss_statistics[label]["timestep"],
                y=error_bars_bounds[label]["ub"],
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                name="Lower bound",
                legendgroup=str(i),
                x=loss_statistics[label]["timestep"],
                y=error_bars_bounds[label]["lb"],
                line=dict(color="rgba(255,255,255,0)"),
                fill="tonexty",
                fillcolor=color_add_alpha(colors_dict[label], error_bars_alpha),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        width=800,
        height=500,
        xaxis_title="Training iteration",
        yaxis_title="Loss",
        # yaxis_tickformat="e",
        yaxis_exponentformat="E",
        margin=dict(l=80, r=10, t=30, b=60),
        legend_traceorder="reversed",
    )

    if hover_compare:
        fig.update_layout(hovermode="x")

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


def activity_heatmap(
    activity: Float[Array, "time unit"],
    colorscale: str = "viridis",
    layout_kws: Optional[dict] = None,
    **kwargs,
) -> go.Figure:
    """Plot activity of all units in a network layer over time, on a single trial.

    !!! Note
        This is a helper for Plotly's [`imshow`](https://plotly.com/python/imshow/),
        when the data is an array of neural network unit activities with shape
        `(time, unit)`.

    !!! Example
        When working with a `SimpleFeedback` model built with a `SimpleStagedNetwork`
        controller—for example, if we've constructed our `model` using
        [`point_mass_nn`][feedbax.xabdeef.models.point_mass_nn]—we can plot the activity
        of the hidden layer of the network:

        ```python
        from feedbax import tree_take

        states = task.eval(model, key=key_eval)  # States for all validation trials.
        states_trial0 = tree_take(states, 0)
        activity_heatmap(states_trial0.net.hidden)
        ```

    Arguments:
        activity: The array of activity over time for each unit in a network layer.
        colorscale: The name of the Plotly [color scale](https://plotly.com/python/builtin-colorscales/)
            to use.
    """
    fig = px.imshow(
        activity.T,
        aspect="auto",
        color_continuous_scale=colorscale,
        labels=dict(x="Time step", y="Unit", color="Activity"),
        **kwargs,
    )
    if layout_kws is not None:
        fig.update_layout(layout_kws)
    return fig


def activity_sample_units(
    activities: Float[Array, "*trial time unit"],
    n_samples: int = 4,
    unit_includes: Optional[Sequence[int]] = None,
    colors: Optional[list[str]] = None,
    row_height: int = 150,
    layout_kws: Optional[dict] = None,
    trial_label: str = "Trial",  # TODO: Rename
    *,
    key: PRNGKeyArray,
    **kwargs,
) -> go.Figure:
    """Plot activity over multiple trials for a random sample of network units.

    The result is a figure with `n_samples + len(unit_includes)` subplots, arranged
    in `cols` columns.

    When this function is called more than once in the course of an analysis, if the
    same `key` is passed and the network layer has the same number of units—that
    is, the last dimension of `activities` has the same size—then the same subset of
    units will be sampled.

    Arguments:
        activities: An array of trial-by-trial activity over time for each unit in a
            network layer.
        n_samples: The number of units to sample from the layer. Along with `unit_includes`,
            this determines the number of subplots in the figure.
        unit_includes: Indices of specific units to include in the plot, in addition to
            the `n_samples` randomly sampled units.
        colors: A list of colors.
        row_height: How tall (in pixels) to make the figure, as a factor of units sampled.
        layout_kws: Additional kwargs with which to update the layout of the figure before
            returning.
        trial_label: The text label for the batch dimension. For example, if `activities`
            gives evaluations across model replicates, we may wish to pass
            `trial_label="Replicate"` to properly label the legend and tooltips.
        key: A random key used to sample the units to plot.
    """

    # Make sure `activities` has shape (trials, time steps, units).
    # If multiple batch dimensions are present, flatten them.
    if len(activities.shape) == 2:
        activities = activities[None, ...]
    elif len(activities.shape) > 3:
        activities = activities.reshape(-1, *activities.shape[-2:])
    elif len(activities.shape) != 3:
        raise ValueError("Invalid shape for ")

    unit_idxs = jr.choice(key, np.arange(activities.shape[-1]), (n_samples,), replace=False)
    if unit_includes is not None:
        unit_idxs = np.concatenate([unit_idxs, np.array(unit_includes)])
    unit_idxs = np.sort(unit_idxs)
    unit_idx_strs = [str(i) for i in unit_idxs]

    xs = np.array(activities[..., unit_idxs])

    # Join all the data into a dataframe.
    df = pl.concat(
        [
            # For each trial, construct all timesteps of x, y data.
            pl.from_numpy(x, schema=unit_idx_strs).hstack(
                pl.DataFrame(
                    {
                        "Timestep": np.arange(x.shape[0]),
                        # Note that "trial" here could be between- or within-condition.
                        trial_label: pl.repeat(i, x.shape[0], eager=True),
                    }
                )
            )
            for i, x in enumerate(xs)
        ]
    ).melt(
        id_vars=["Timestep", trial_label],
        value_name="Activity",
        variable_name="Unit",
    )

    fig = px.line(
        df,
        x="Timestep",
        y="Activity",
        color=trial_label,
        facet_row="Unit",
        color_discrete_sequence=colors,
        height=row_height * len(unit_idxs),
        **kwargs,
    )

    # Replace multiple y-axis labels with a single one.
    fig.for_each_yaxis(lambda y: y.update(title=""))
    fig.add_annotation(
        x=-0.07,
        y=0.5,
        text="Activity",
        textangle=-90,
        showarrow=False,
        font=dict(size=14),
        xref="paper",
        yref="paper",
    )

    # fig.update_yaxes(zerolinewidth=2, zerolinecolor='rgb(200,200,200)')
    fig.update_yaxes(zerolinewidth=0.5, zerolinecolor="black")

    # Improve formatting of subplot "Unit=" labels.
    fig.for_each_annotation(
        lambda a: a.update(
            text=a.text.replace("=", " "),
            font=dict(size=14),
        )
    )

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


def tree_of_2d_timeseries_to_df(
    tree: PyTree[Shaped[Array, "condition timestep xy=2"], "T"],
    labels: Optional[PyTree[str, "T"]] = None,
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


def effector_trajectories(
    states: SimpleFeedbackState | PyTree[Float[Array, "trial time ..."] | Any],
    where_data: Optional[
        Callable[[PyTree[Array]], Sequence[Shaped[Array, "*batch trial time xy=2"], ...]]
    ] = None,
    var_labels: Optional[tuple[str, ...]] = None,
    step: int = 1,  # plot every step-th trial
    trial_specs: Optional["TaskTrialSpec"] = None,
    endpoints: Optional[tuple[Float[Array, "trial xy=2"], Float[Array, "trial xy=2"]]] = None,
    straight_guides: bool = False,
    workspace: Optional[Float[Array, "bounds=2 xy=2"]] = None,
    cmap_name: Optional[str] = None,
    colors: Optional[list[str]] = None,
    color: Optional[str | tuple[float, ...]] = None,
    mode: str = "markers+lines",
    ms: int = 5,
    ms_init: int = 12,
    ms_goal: int = 12,
    control_labels: Optional[tuple[str, str, str]] = None,
    control_label_type: str = "linear",
    layout_kws: Optional[dict] = None,
    trace_kwargs: Optional[dict] = None,
    **kwargs,
):
    """Plot trajectories of position, velocity, network output.

    Arguments:
        states: A model state or PyTree of arrays from which the variables to be
            plotted can be extracted.
        where_data: If `states` is provided as an arbitrary PyTree of arrays,
            this function should be provided to extract the relevant arrays.
            It should take `states` and return a tuple of arrays.
        var_labels: Labels for the variables selected by `where_data`.
        step: Plot every `step`-th trial. This is useful when `states` contains
            information about a very large set of trials, and we only want to
            plot a subset of them.
        trial_specs: The specifications for the trials being plotted. If supplied,
            this is used to plot markers at the initial and goal positions.
        endpoints: The initial and goal positions for the trials being plotted.
            Overrides `trial_specs`.
        straight_guides: If this is `True` and `endpoints` are provided, straight
            dashed lines will be drawn between the initial and goal positions.
        workspace: The workspace bounds. If provided, the bounds are drawn as a
            rectangle.
        cmap_name: The name of the Matplotlib [colormap](https://matplotlib.org/stable/gallery/color/colormap_reference.html)
            to use across trials.
        colors: A sequence of colors, one for each plotted trial. Overrides `cmap_name`.
        color: A single color to use for all trials. Overrides `cmap_name` but not `colors`.
        ms: Marker size for plots of states (trajectories).
        ms_source: Marker size for the initial position, if `trial_specs`/`endpoints`
            is provided.
        ms_target: Marker size for the goal position.
        control_label_type: If `'linear'`, labels the final (controller output/force)
            plot as showing Cartesian forces. If `'torques'`, labels the plot as showing
            the torques of a two-segment arm.
        control_labels: A tuple giving the labels for the title, x-axis, and y-axis
            of the final (controller output/force) plot. Overrides `control_label_type`.
    """
    var_labels_ = var_labels

    vars_tuple: tuple[Shaped[Array, "*batch trial time xy=2"], ...]
    if where_data is not None:
        vars_tuple = tuple(where_data(states))
        if var_labels is None:
            var_labels = where_func_to_attr_str_tree(where_data)

    elif isinstance(states, SimpleFeedbackState):
        vars_tuple = (
            states.mechanics.effector.pos,
            states.mechanics.effector.vel,
            states.efferent.output,
        )
        if var_labels is None:
            var_labels_ = ("Position", "Velocity", "Force")
    else:
        raise ValueError(
            "If `states` is not a `SimpleFeedbackState`, `where_data` must be provided."
        )

    if len(vars_tuple[0].shape) > 3:
        # Collapse to a single batch dimension
        vars_tuple = jt.map(
            lambda arr: np.reshape(arr, (-1, *arr.shape[-3:])),
            vars_tuple,
        )
        dfs = [
            tree_of_2d_timeseries_to_df(v, labels=var_labels_)
            for v in zip(*jt.map(tuple, vars_tuple))
        ]
        dfs = [
            df.hstack(pl.DataFrame({"Trial": pl.repeat(i, len(df), eager=True)}))
            for i, df in enumerate(dfs)
        ]
        df = pl.concat(dfs, how="vertical")
    else:
        df = tree_of_2d_timeseries_to_df(
            vars_tuple,
            labels=var_labels_,
        )

    n_vars = df["var"].n_unique()

    # if cmap_name is None:
    #     if positions.shape[0] < 10:
    #         cmap_name = "tab10"
    #     else:
    #         cmap_name = "viridis"

    # TODO: Use `go.Figure` for more control over trial vs. condition, in batched case
    # fig = go.Figure()
    # fig.add_traces()
    # TODO: Separate control/indexing of lines vs. markers; e.g. thin line-only traces,
    # plus markers only at the end of the reach

    if colors is None:
        n_conditions = vars_tuple[0].shape[-3]
        colors = [str(c) for c in sample_colorscale("phase", n_conditions + 1)]

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=df["Condition"].cast(pl.String),
        facet_col="var",
        facet_col_spacing=0.05,
        color_discrete_sequence=colors,
        labels=dict(color="Condition"),
        custom_data=["Condition", "var", "Timestep"],
        **kwargs,
    )

    if trace_kwargs is None:
        trace_kwargs = {}

    fig.for_each_trace(
        lambda trace: trace.update(
            mode=mode,
            hovertemplate=(
                "Condition: %{customdata[0]}<br>"
                "Time step: %{customdata[2]}<br>"
                "x: %{x:.2f}<br>"
                "y: %{y:.2f}<br>"
                "<extra></extra>"
            ),
            **trace_kwargs,
        )
    )

    fig.update_traces(marker_size=ms)

    if endpoints is not None:
        endpoints_arr = np.array(endpoints)  # type: ignore
    else:
        if trial_specs is not None:
            target_specs = trial_specs.targets["mechanics.effector.pos"]
            if target_specs.value is not None:
                endpoints_arr = np.array(  # type: ignore
                    [
                        trial_specs.inits["mechanics.effector"].pos,
                        target_specs.value[:, -1],
                    ]
                )
            else:
                endpoints_arr = None
        else:
            endpoints_arr = None

    if endpoints_arr is not None:
        colors = [d.marker.color for d in fig.data[::n_vars]]

        n_trials = endpoints_arr.shape[1]

        # Add init and goal markers
        for j, (label, (ms, symbol)) in enumerate(
            {
                "Init": (ms_init, "square"),
                "Goal": (ms_goal, "circle"),
            }.items()
        ):
            fig.add_traces(
                [
                    go.Scatter(
                        name=f"{label} {i}",
                        meta=dict(label=label),
                        legendgroup=label,
                        hovertemplate=f"{label} {i}<extra></extra>",
                        x=endpoints_arr[j, i, 0][None],
                        y=endpoints_arr[j, i, 1][None],
                        mode="markers",
                        marker=dict(
                            size=ms,
                            symbol=symbol,
                            color="rgba(255, 255, 255, 0)",
                            line=dict(
                                color=colors[i],
                                width=2,
                            ),
                        ),
                        xaxis="x1",
                        yaxis="y1",
                        # TODO: Show once in legend, for all markers of type j
                        showlegend=i < 1,
                    )
                    for i in range(n_trials)
                ]
            )

        # Add dashed straight lines from init to goal.
        if straight_guides:
            fig.add_traces(
                [
                    go.Scatter(
                        x=endpoints_arr[:, i, 0],
                        y=endpoints_arr[:, i, 1].T,
                        mode="lines",
                        line_dash="dash",
                        line_color=colors[i],
                        showlegend=False,
                        xaxis="x1",
                        yaxis="y1",
                    )
                    for i in range(endpoints_arr.shape[1])
                ]
            )

    # Constrain the axes of each subplot to be scaled equally.
    # That is, keep square aspect ratios.
    fig.update_layout(
        {f"yaxis{i}": dict(scaleanchor=f"x{i}") for i in [""] + list(range(2, n_vars + 1))}
    )

    # Omit the "var=" part of each subplot title
    fig.for_each_annotation(
        lambda a: a.update(
            text=a.text.split("=")[-1],
            font=dict(size=14),
        )
    )

    # Facet plot shares axes by default.
    unshare_axes(fig)

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


def sample_colorscale_unique(colorscale, samplepoints: int, **kwargs):
    """Helper to ensure we don't get repeat colors when using cyclical colorscales.

    Also avoids the division-by-zero error that `sample_colorscale` raises when `samplepoints == 1`.
    """
    colors = plc.get_colorscale(colorscale)
    if samplepoints == 1:
        n_sample = 2
        idxs = slice(1, None)
    elif colors[0][1] == colors[-1][1]:
        n_sample = samplepoints + 1
        idxs = slice(None, -1)
    else:
        n_sample = samplepoints
        idxs = slice(None)

    return sample_colorscale(colorscale, n_sample, **kwargs)[idxs]


def arr_to_nested_tuples(arr):
    """Like `ndarray.tolist()` but ensures the bottom-most level is tuples."""
    if arr.ndim == 1:
        return tuple(arr.tolist())
    elif arr.ndim > 1:
        return [arr_to_nested_tuples(sub_arr) for sub_arr in arr]
    else:
        return arr.item()  # For 0-dimensional arrays


def arr_to_rgb(arr):
    return f"rgb({', '.join(map(str, arr))})"


def adjust_color_brightness(colors, factor=0.8):
    colors_arr = np.array(plc.convert_colors_to_same_type(colors, colortype="tuple")[0])
    return list(map(arr_to_rgb, factor * colors_arr))


def is_trace(element):
    return isinstance(element, BaseTraceType)


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


def _compute_colors(
    base_shape: tuple[int, ...],
    colors,
    colorscale: str,
    colorscale_axis: int,
    stride: int,
):
    """Return (color_sequence[C,3], colors_broadcast[*batch,3], color_idxs[*batch], default_labels, idxs_or_None).
    Works on the *per-leaf* shape (*batch, time, d). Also returns the indices to slice along
    the colorscale axis when `stride != 1`.
    """
    batch_shape = base_shape[:-2]  # *batch
    if colorscale_axis < 0:
        colorscale_axis += len(base_shape)
    if colorscale_axis >= len(base_shape) - 2:
        raise ValueError(f"colorscale_axis {colorscale_axis} points to a non-batch dimension")

    C_full = base_shape[colorscale_axis]
    if stride != 1:
        idxs = np.arange(C_full)[::stride]
        C = len(idxs)
    else:
        idxs = None
        C = C_full

    # Determine color sequence for C groups
    if colors is None:
        color_sequence = np.array(
            sample_colorscale_unique(colorscale, C, colortype="tuple")
        )  # (C,3)
    elif isinstance(colors, str):
        converted, _ = convert_colors_to_same_type([colors])
        color_sequence = np.array(converted[0] * C).reshape(C, -1)
    else:
        if len(colors) != C:
            raise ValueError("Length of colors must match number of color groups (after stride)")
        converted, _ = convert_colors_to_same_type(list(colors))
        color_sequence = np.array(converted)

    # Build per-group color indices and broadcasted RGBs across other batch dims
    color_idxs = np.arange(C).reshape((C,) + (1,) * (len(batch_shape) - 1))
    full_shape = (C,) + tuple(
        batch_shape[i] for i in range(len(batch_shape)) if i != colorscale_axis
    )
    color_idxs = np.broadcast_to(color_idxs, full_shape)

    colors_broadcast = np.broadcast_to(
        np.expand_dims(
            color_sequence, axis=tuple(1 + np.arange(len(full_shape) - 1))
        ),  # (C,1,1,...,3)
        full_shape + (3,),
    )

    default_labels = list(range(C))
    return color_sequence, colors_broadcast, color_idxs, default_labels, idxs


def _prepare_leaf_arrays(
    x: np.ndarray,
    colors_broadcast,
    color_idxs,
    colorscale_axis: int,
    n_curves_max: Optional[int],
):
    """Given one leaf array x with shape (*batch, time, d), produce flattened curve arrays.
    Returns: var_flat[NC, time, d], colors_flat[NC,3], color_idxs_flat[NC]"""
    full_ndim = x.ndim
    axis0 = colorscale_axis if colorscale_axis >= 0 else full_ndim + colorscale_axis

    # move colorscale axis to front and collapse the rest into a single axis of curves
    x = np.reshape(np.moveaxis(x, axis0, 0), (x.shape[axis0], -1, *x.shape[-2:]))  # (C, K, time, d)
    cb = np.reshape(colors_broadcast, (colors_broadcast.shape[0], -1, 3))  # (C,K,3)
    ci = np.reshape(color_idxs, (color_idxs.shape[0], -1))  # (C,K)

    # subsample per-group curves if requested
    K = x.shape[1]
    if n_curves_max is not None and K > n_curves_max:
        idxs_sample = np.random.choice(np.arange(K), n_curves_max, replace=False).astype(int)
        x = x[:, idxs_sample]
        cb = cb[:, idxs_sample]
        ci = ci[:, idxs_sample]

    # collapse groups and curves to a single curve axis
    var_flat = np.reshape(x, (-1, *x.shape[-2:]))  # (C*K, time, d)
    colors_flat = np.reshape(cb, (-1, 3))  # (C*K,3)
    color_idxs_flat = np.reshape(ci, (-1,))  # (C*K,)

    return var_flat, colors_flat, color_idxs_flat


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


StrMapping: TypeAlias = Mapping[str, Any]


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


def _normalize_collection_arg[S](
    seq: SeqOf[Any],  # the reference subplots_data (now a SeqOf)
    seq_label: str,  # label used in error messages
    value: S | SeqOf[S] | None,  # scalar S, per-leaf SeqOf[S], or None
    arg_label: str,  # name of the argument (for errors)
    default_fn: Callable[[SeqOf[Any]], list[S | None]],
    *,
    is_leaf: Callable[[Any], TypeGuard[S]] | None = None,
    leaf_type: type | None = None,  # used only if is_leaf is None
) -> list[S | None]:
    """Normalize an argument that may be a single leaf value S, a per-leaf sequence
    (list/tuple/ValuesView) of S, or None (which triggers default_fn).

    - Mappings are not accepted as per-leaf containers; if a Mapping is a valid *leaf*,
      supply a custom `is_leaf` TypeGuard that recognizes it.
    """
    # Build a predicate that narrows to S
    if is_leaf is None:
        if leaf_type is None:
            raise TypeError(f"{arg_label}: leaf_type required when is_leaf is None")

        def _is_leaf(x: Any, /) -> TypeGuard[S]:
            return isinstance(x, leaf_type)
    else:
        _is_leaf = is_leaf

    n = len(seq)

    # None -> defaults
    if value is None:
        return default_fn(seq)

    # Single leaf S -> replicate
    if _is_leaf(value):
        return [value] * n

    # Per-leaf sequence (only list/tuple/ValuesView)
    if isinstance(value, (list, tuple, ValuesView)):
        out: list[S | None] = list(value)
        if len(out) != n:
            raise ValueError(f"{arg_label} length must match {seq_label} length")
        for v in out:
            if not _is_leaf(v):
                raise TypeError(
                    f"{arg_label} must be a sequence containing only the expected leaf type"
                )
        return out

    # Everything else is invalid under SeqOf contract
    raise TypeError(
        f"{arg_label} must be a single value, a list/tuple/ValuesView of per-leaf values, or None"
    )


TRAJ_SCATTER_KWS_DEFAULT: StrMapping = MappingProxyType(
    dict(
        mode="lines",
        marker_size=5,
    )
)


def trajectories[T](
    subplots_data: SeqOfT[Float[Array, "... time dims"] | np.ndarray, T],
    colorscale_axis: int = 0,  # which batch axis forms color groups
    scatter_kws: (StrMapping | SeqOfT[StrMapping, T]) = TRAJ_SCATTER_KWS_DEFAULT,
    subplot_titles: Optional[SeqOfT[str, T]] = None,
    axes_labels: Optional[AxesLabels | SeqOfT[AxesLabels, T]] = None,
    show_mean: bool = True,
    legend_title: Optional[str] = None,
    legend_labels: Optional[Sequence] = None,
    colors: Optional[str | Sequence[str]] = None,
    colorscale: str = "phase",
    stride: int = 1,
    n_curves_max: Optional[int] = None,
    endpoints: Literal["all", "mean", "none"] = "mean",
    start_marker_kws: Optional[StrMapping] = None,
    end_marker_kws: Optional[StrMapping] = None,
    lighten_mean: float = 0.8,
    mean_exclude_axes: Sequence[int] = (),
    mean_scatter_kws: Optional[StrMapping] = None,
    master_axes_labels_2d: AxesLabels = AxesLabels(None, None),
    shared_yaxes_label: bool = True,
    layout_kws: Optional[Mapping[str, Any]] = None,
    padding_factor: float = 0.1,
):
    """Unified 2D/3D trajectory plotting with subplot-per-leaf.

    - `vars_` can be any (possibly nested) Collection; we traverse with `jax.tree.map`/`leaves` so dicts yield values.
    - Detects 2D vs 3D per subplot from the last axis size (2 or 3).
    - Supports multiple batch dims; one is selected by `colorscale_axis` to define legend/color groups.
    - Other batch dims are aggregated into individual curves per group, with optional subsampling via `n_curves_max`.
    - Distinct start/end endpoint markers for both 2D and 3D.
    - Optional mean trajectories per color group (works for 2D & 3D), with axis-exclusion control.
    - `axes_labels` and `scatter_kws` accept either single values or a Collection matching the number of leaves.
    """
    default_start_marker_kws = dict(
        symbol=None,
        size=10,
        line_width=2,
    )
    default_end_marker_kws = dict(
        symbol="circle-open",
        size=6,
        line_width=2,
    )

    if start_marker_kws is None:
        start_marker_kws = {}
    if end_marker_kws is None:
        end_marker_kws = {}

    endpoints_spec: dict[str, tuple[slice, dict[str, Any]]] = dict(
        start=(
            slice(0, 1),
            default_end_marker_kws | dict(end_marker_kws),
        ),
        end=(
            slice(-1, None),
            default_start_marker_kws | dict(start_marker_kws),
        ),
    )

    if mean_scatter_kws is None:
        mean_scatter_kws = {}
    mean_scatter_kws = dict(mean_scatter_kws)

    vars_norm = _normalize_vars(subplots_data)
    subplots_data_list = list(vars_norm)
    if not all(isinstance(x, (Array, np.ndarray)) for x in subplots_data_list):
        raise TypeError(
            "`subplots_data` must be a flat sequence-like container of JAX or numpy arrays."
        )

    subplot_titles_list = _normalize_collection_arg(
        subplots_data,
        "subplots_data",
        subplot_titles,
        arg_label="subplot_titles",
        #! The `leaves(tree_labels(...))` here will always just be range(len(vars_))
        #! so long as we're only dealing in sequence-likes
        default_fn=lambda vars_: [f"<b>{label}</b>" for label in jt.leaves(tree_labels(vars_))],
        leaf_type=str,
    )

    axes_labels_list = _normalize_collection_arg(
        subplots_data,
        "subplots_data",
        axes_labels,
        arg_label="axes_labels",
        default_fn=lambda vars_: [
            AxesLabels("x", "y", "z" if arr.shape[-1] == 3 else None) for arr in vars_
        ],
        leaf_type=AxesLabels,
    )

    scatter_kws_list = _normalize_collection_arg(
        subplots_data,
        "subplots_data",
        scatter_kws,
        arg_label="scatter_kws",
        default_fn=lambda vars_: [{}] * len(vars_),
        leaf_type=Mapping,
        is_leaf=_is_str_mapping,
    )

    # Build subplot specs by leaf dim
    types = ["scene" if arr.shape[-1] == 3 else "xy" for arr in subplots_data_list]
    specs = [[{"type": t} for t in types]]

    fig = make_subplots(
        rows=1,
        cols=len(types),
        specs=specs,
        subplot_titles=subplot_titles_list,
        horizontal_spacing=0.1,
        x_title=master_axes_labels_2d.x,
        y_title=master_axes_labels_2d.y,
    )

    # Shared 2D y-label handling
    if shared_yaxes_label:
        for col_idx, t in enumerate(types, start=1):
            if t == "xy":
                fig.update_yaxes(title_text="", row=1, col=col_idx)

    # Constrain 2D subplots to square aspect
    for i, t in enumerate(types, start=1):
        if t == "xy":
            d = "" if i == 1 else i
            fig.update_layout({f"yaxis{d}": dict(scaleanchor=f"x{d}")})

    # Map from 3D subplot order to scene id (scene, scene2, ...)
    scene_indices = {}
    scene_count = 0
    for i, t in enumerate(types, start=1):
        if t == "scene":
            scene_count += 1
            scene_indices[i] = "" if scene_count == 1 else str(scene_count)

    # Only show legends on the first subplot to avoid duplication
    def _legend_on_this_leaf(leaf_idx: int, show_mean: bool, group_first_seen: set, group_idx: int):
        if leaf_idx != 0 or show_mean:
            return False
        if group_idx in group_first_seen:
            return False
        group_first_seen.add(group_idx)
        return True

    group_first_seen: set[int] = set()

    # Iterate over leaves and plot
    for leaf_idx, var in enumerate(subplots_data_list):
        d = var.shape[-1]
        col = leaf_idx + 1
        axlbl = axes_labels_list[leaf_idx]
        skws: Mapping[str, Any] = scatter_kws_list[leaf_idx] or {}

        # Per-leaf coloring setup
        (
            color_sequence,
            colors_broadcast,
            color_idxs,
            default_legend_labels,
            stride_idxs,
        ) = _compute_colors(var.shape, colors, colorscale, colorscale_axis, stride)
        # If stride requested, slice the data along the colorscale axis so groups match colors
        if stride_idxs is not None:
            axis0 = colorscale_axis if colorscale_axis >= 0 else var.ndim + colorscale_axis
            var = np.take(var, stride_idxs, axis=axis0)

        # Legend labels per-leaf
        if legend_labels is None:
            legend_labels_local = list(default_legend_labels)
        else:
            legend_labels_local = list(legend_labels)
            if len(legend_labels_local) != color_sequence.shape[0]:
                legend_labels_local = list(default_legend_labels)

        # Prepare curves for this leaf
        var_flat, colors_flat, color_idxs_flat = _prepare_leaf_arrays(
            var, colors_broadcast, color_idxs, colorscale_axis, n_curves_max
        )

        # Convert RGB tuples to color strings
        colors_rgb_tuples = arr_to_nested_tuples(colors_flat)
        colors_rgb = jt.map(
            lambda x: convert_colors_to_same_type(x)[0][0],
            colors_rgb_tuples,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        is_2d = d == 2
        ts = np.arange(var.shape[-2])

        def _customdata_ts(label):
            return np.concatenate(
                [
                    ts[:, None],
                    np.broadcast_to([[label]], (ts.shape[0], 1)),
                ],
                axis=-1,
            )

        def _add_curve_trace(curve, color_str, label, showlegend_flag):
            common = dict(
                name=label,
                showlegend=showlegend_flag,
                line_color=color_str,
                marker_color=color_str,
                legendgroup=str(label),
            ) | dict(skws)
            if is_2d:
                trace = go.Scatter(
                    **common,
                    x=curve[..., 0],
                    y=curve[..., 1],
                    customdata=_customdata_ts(label),
                )
            else:
                trace = go.Scatter3d(
                    **common,
                    x=curve[:, 0],
                    y=curve[:, 1],
                    z=curve[:, 2],
                    customdata=ts[:, None],
                )
            fig.add_trace(trace, row=1, col=col)

        def _add_endpoint_marker(curve, color_str, label, which: str):
            if endpoints != "all":
                return
            idxs, marker_kws = endpoints_spec[which]
            kwargs = (
                dict(
                    name=f"{label} {which}",
                    showlegend=False,
                    marker_color=color_str,
                    marker_line_color=color_str,
                    mode="markers",
                )
                | marker_kws
            )

            if is_2d:
                trace = go.Scatter(x=curve[idxs, 0], y=curve[idxs, 1], **kwargs)
            else:
                trace = go.Scatter3d(
                    x=curve[idxs, 0],
                    y=curve[idxs, 1],
                    z=curve[idxs, 2],
                    **kwargs,
                )

            fig.add_trace(trace, row=1, col=col)

        # --- per-curve plotting ---
        for i_curve, curve in enumerate(var_flat):
            group_idx = int(color_idxs_flat[i_curve])
            label = legend_labels_local[group_idx]
            showlegend = _legend_on_this_leaf(leaf_idx, show_mean, group_first_seen, group_idx)
            color_str = colors_rgb[i_curve]
            _add_curve_trace(curve, color_str, label, showlegend)
            _add_endpoint_marker(curve, color_str, label, "start")
            _add_endpoint_marker(curve, color_str, label, "end")

        # 2D axis padding
        if is_2d:
            all_y = var_flat[..., 1].flatten()
            if not np.all(np.isnan(all_y)):
                y_min = np.nanmin(all_y)
                y_max = np.nanmax(all_y)
                y_rng = y_max - y_min
                pad = y_rng * padding_factor if y_rng > 0 else 0.1
                # fig.update_yaxes(autorange=True, row=1, col=col)
                fig.update_yaxes(
                    domain=[None, None],
                    range=[y_min - pad, y_max + pad],
                    row=1,
                    col=col,
                )
                fig.update_xaxes(autorange=True, row=1, col=col)

        # mean trajectories (per-leaf)
        if show_mean:
            axis0 = colorscale_axis if colorscale_axis >= 0 else var.ndim + colorscale_axis
            mean_arr = _mean_over_axes(var, axis0, mean_exclude_axes)  # (C, M, time, d)
            for j in range(mean_arr.shape[0]):  # color groups
                label = legend_labels_local[j]
                for k in range(mean_arr.shape[1]):  # preserved batch combos
                    mean_color = arr_to_rgb(lighten_mean * np.array(color_sequence[j]))
                    kwargs = (
                        dict(
                            name=label,
                            mode="lines",
                            line_color=mean_color,
                            showlegend=(leaf_idx == 0 and k == 0),
                        )
                        | mean_scatter_kws
                    )
                    xy = mean_arr[j, k]
                    if is_2d:
                        trace = go.Scatter(
                            x=xy[..., 0], y=xy[..., 1], customdata=_customdata_ts(label), **kwargs
                        )
                    else:
                        trace = go.Scatter3d(
                            x=xy[:, 0], y=xy[:, 1], z=xy[:, 2], customdata=ts[:, None], **kwargs
                        )

                    fig.add_trace(trace, row=1, col=col)

                    if endpoints == "mean":
                        for endpt_label, (
                            idxs,
                            marker_kws,
                        ) in endpoints_spec.items():
                            kwargs = dict(
                                mode="markers",
                                showlegend=False,
                                name=f"{label} {endpt_label}",
                            )

                            marker_kwargs = (
                                dict(
                                    line_color=mean_color,
                                    color=mean_color,
                                )
                                | marker_kws
                            )

                            if is_2d:
                                fig.add_trace(
                                    go.Scatter(
                                        x=xy[idxs, 0],
                                        y=xy[idxs, 1],
                                        marker=marker_kwargs,
                                        **kwargs,
                                    ),
                                    row=1,
                                    col=col,
                                )
                            else:
                                fig.add_trace(
                                    go.Scatter3d(
                                        x=xy[idxs, 0],
                                        y=xy[idxs, 1],
                                        z=xy[idxs, 2],
                                        marker=marker_kwargs,
                                        **kwargs,
                                    ),
                                    row=1,
                                    col=col,
                                )

        # per-subplot axis titles
        if axlbl is not None:
            if axlbl.z is None:
                fig.update_xaxes(title_text=axlbl.x, row=1, col=col)
                fig.update_yaxes(title_text=axlbl.y, row=1, col=col)
            else:
                scene_suffix = scene_indices[col]
                fig.update_layout(
                    {
                        f"scene{scene_suffix}": dict(
                            xaxis_title=axlbl.x,
                            yaxis_title=axlbl.y,
                            zaxis_title=axlbl.z,
                        )
                    }
                )

    # Hover templates (keep minimal to avoid mismatched customdata)
    def _apply_hover(trace: BaseTraceType):
        if isinstance(trace, go.Scatter) and getattr(trace, "customdata", None) is not None:
            trace.update(
                hovertemplate=(
                    "Time step: %{customdata[0]}<br>x: %{x:.2f}<br>y: %{y:.2f}<br><extra></extra>"
                )
            )
        elif isinstance(trace, go.Scatter3d) and getattr(trace, "customdata", None) is not None:
            trace.update(
                hovertemplate=(
                    "Time step: %{customdata[0]}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br><extra></extra>"
                )
            )

    fig.for_each_trace(_apply_hover)

    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


# Backwards compatible wrappers
def trajectories_2D[T](
    vars_: SeqOfT[Float[Array, "*trial time dims"], T] | SeqOfT[np.ndarray, T],
    var_labels: Optional[SeqOfT[str, T]] = None,
    axes_labels: Optional[AxesLabels | SeqOfT[AxesLabels, T]] = None,
    var_endpoint_ms: int = 0,
    show_mean: bool = True,
    lighten_mean: float = 0.8,
    colors: Optional[Float[Array, "*trial rgb=3"] | str] = None,
    colorscale_axis: int = 0,
    colorscale: str = "phase",
    stride: int = 1,
    n_curves_max: Optional[int] = None,
    legend_title: Optional[str] = None,
    legend_labels: Optional[Sequence] = None,
    curves_mode: str = "lines",
    ms: int = 5,
    master_axes_labels: AxesLabels = AxesLabels(None, None),
    shared_yaxes_label: bool = True,
    layout_kws: Optional[dict] = None,
    scatter_kws: Optional[dict] = None,
    mean_scatter_kws: Optional[dict] = None,
    mean_exclude_axes: Sequence[int] = (),
    padding_factor: float = 0.1,
    **_,
):
    endpoints = "all" if (var_endpoint_ms or 0) > 0 else "none"
    end_marker_kws = dict(symbol="circle-open", size=var_endpoint_ms or 6, line_width=2)

    if scatter_kws is None:
        scatter_kws = {}
    scatter_kws |= dict(
        mode=curves_mode,
        marker_size=ms,
    )

    if isinstance(colors, Array):
        colors_ = list(colors)
    else:
        colors_ = colors

    return trajectories(
        vars_,
        subplot_titles=var_labels,
        legend_title=legend_title,
        legend_labels=legend_labels,
        colors=colors_,
        colorscale_axis=colorscale_axis,
        colorscale=colorscale,
        stride=stride,
        n_curves_max=n_curves_max,
        endpoints=endpoints,
        end_marker_kws=end_marker_kws,
        show_mean=show_mean,
        lighten_mean=lighten_mean,
        mean_exclude_axes=mean_exclude_axes,
        mean_scatter_kws=mean_scatter_kws,
        axes_labels=axes_labels,
        master_axes_labels_2d=master_axes_labels,
        shared_yaxes_label=shared_yaxes_label,
        layout_kws=layout_kws,
        scatter_kws=scatter_kws,
        padding_factor=padding_factor,
    )


def trajectories_3D(
    traj: Float[Array | np.ndarray, "trials time 3"],
    endpoint_symbol: Optional[str] = "circle-open",
    start_symbol: Optional[str] = None,
    fig: Optional[go.Figure] = None,
    colors: str | Sequence[str] | None = None,
    axis_labels: Optional[tuple[str, str, str]] = ("x", "y", "z"),
    mode: str = "lines",
    name: Optional[str] = "State trajectory",
    colorscale_axis: int = 0,
    colorscale: str = "phase",
    stride: int = 1,
    n_curves_max: Optional[int] = None,
    show_mean: bool = False,
    **kwargs,
):
    vars_ = [traj]

    endpoints = "all" if (endpoint_symbol is not None or start_symbol is not None) else "none"
    if start_symbol is None:
        start_marker_kws = None
    else:
        start_marker_kws = dict(symbol=start_symbol)
    if endpoint_symbol is None:
        end_marker_kws = None
    else:
        end_marker_kws = dict(symbol=endpoint_symbol)

    if axis_labels is None:
        axes_labels = None
    else:
        axes_labels = AxesLabels(*axis_labels)

    fig = trajectories(
        vars_,
        subplot_titles=[name] if name is not None else [""],
        legend_title=name or "Condition",
        legend_labels=None,
        colors=colors,
        colorscale_axis=colorscale_axis,
        colorscale=colorscale,
        stride=stride,
        n_curves_max=n_curves_max,
        start_marker_kws=start_marker_kws,
        end_marker_kws=end_marker_kws,
        endpoints=endpoints,
        show_mean=show_mean,
        axes_labels=axes_labels,
        layout_kws=(kwargs.get("layout_kws") or {}),
        scatter_kws={k: v for k, v in kwargs.items() if k not in {"layout_kws"}},
    )
    return fig if fig is not None else go.Figure(fig)


def plot_eigvals(
    eigvals: Float[Array, "batch eigvals"],
    colors: str | Sequence[str] | None = None,
    colorscale: str = "phase",
    labels: Optional[Sequence[Optional[str]]] = None,
    mode: str = "markers",
    marker_size: int = 5,
    fig: Optional[go.Figure] = None,
    layout_kws: Optional[dict] = None,
    **kwargs,
):
    """Plot eigenvalues inside a unit circle with dashed axes."""
    if fig is None:
        fig = go.Figure(
            layout=dict(
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )
        )

    if colors is not None:
        if isinstance(colors, str):
            pass
        elif len(colors) == np.prod(eigvals.shape):
            pass
        elif len(colors) == eigvals.shape[0]:
            colors = np.repeat(
                np.array(colors),
                np.prod(eigvals.shape[1:]),
            )

    # Add a unit circle
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-1,
        y0=-1,
        x1=1,
        y1=1,
        line_color="black",
    )

    # Add dashed axes lines
    fig.add_hline(0, line_dash="dot", line_color="grey")
    fig.add_vline(0, line_dash="dot", line_color="grey")

    if labels is None:
        labels = [None] * len(eigvals)

    # Plot eigenvalues
    fig.add_trace(
        go.Scatter(
            x=np.real(np.ravel(eigvals)),
            y=np.imag(np.ravel(eigvals)),
            mode=mode,
            marker_size=marker_size,
            marker_color=colors,
            marker_colorscale=colorscale,
            **kwargs,
        )
    )

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig
