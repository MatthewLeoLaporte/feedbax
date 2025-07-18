"""

:copyright: Copyright 2024 by Matt L. Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

from collections import OrderedDict
from collections.abc import Callable, Sequence
from itertools import zip_longest
import logging
from math import sqrt
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar

import equinox as eqx
from feedbax._tree import apply_to_filtered_leaves, tree_infer_batch_size, tree_prefix_expand, tree_take, tree_unzip
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Shaped
import numpy as np
# pyright: reportMissingTypeStubs=false
from plotly.basedatatypes import BaseTraceType
import plotly.colors as plc
from plotly.colors import convert_colors_to_same_type, sample_colorscale
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import polars as pl

from feedbax import tree_labels
from feedbax._tree import tree_zip
from feedbax.bodies import SimpleFeedbackState
from feedbax.loss import LossDict
from feedbax.misc import where_func_to_attr_str_tree, is_none

if TYPE_CHECKING:
    from feedbax.task import TaskTrialSpec
    from feedbax.train import TaskTrainerHistory


logger = logging.getLogger(__name__)


T = TypeVar('T')


pio.templates.default = "plotly_white"
DEFAULT_COLORS = pio.templates[pio.templates.default].layout.colorway  # pyright: ignore


def color_add_alpha(rgb_str: str, alpha: float):
    return f"rgba{rgb_str[3:-1]}, {alpha})"


def columns_mean_std(dfs: PyTree[pl.DataFrame], index_col: Optional[str] = None):
    if index_col is not None:
        spec = {
            index_col: pl.col(index_col),
            "mean": pl.concat_list(pl.col('*').exclude(index_col)).list.mean(),
            "std": pl.concat_list(pl.col('*').exclude(index_col)).list.std(),
        }
    else:
        spec = {
            "mean": pl.concat_list(pl.col('*')).list.mean(),
            "std": pl.concat_list(pl.col('*')).list.std(),
        }

    return jax.tree_map(
        lambda df: df.select(**spec),
        dfs,
    )


def errorbars(col_means_stds: PyTree[pl.DataFrame], n_std: int):
    return jax.tree_map(
        lambda df: df.select(
            lb=pl.col('mean') - n_std * pl.col('std'),
            ub=pl.col('mean') + n_std * pl.col('std'),
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
        labels=dict(index="Time step", value=var_label, variable='Trial'),
        **kwargs,
    )
    if layout_kws is not None:
        fig.update_layout(layout_kws)
    return fig


def profiles(
    vars_: PyTree[Float[Array, "*batch timestep"], "T"],
    keep_axis: Optional[PyTree[int, "T ..."]] = None,
    mode: Literal["std", "curves"] = "std",
    stride_curves: int = 1,
    timesteps: Optional[PyTree[Float[Array, "timestep"], "T"]] = None,
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
    fig: Optional[go.Figure] = None,
) -> go.Figure:
    """Plot 1D state profiles as lines with standard deviation bars.

    `keep_axis` will retain one dimension of data, and plot one mean+/-std curve for each entry in that axis
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
            batch_axes, tree_prefix_expand(keep_axis, vars_),
            is_leaf=lambda x: isinstance(x, tuple) and eqx.is_array_like(x[0]),
        )

    means = jt.map(
        lambda x, axis: np.nanmean(x, axis=axis),
        vars_, mean_axes,
    )

    stds = jt.map(
        lambda x, axis: np.nanstd(x, axis=axis),
        vars_, mean_axes,
    )

    if keep_axis is None:
        means, stds = jt.map(lambda arr: arr[None, ...], (means, stds))
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
    colors_rgb, _ = convert_colors_to_same_type(colors, colortype='rgb')  # type: ignore

    if hline is not None:
        fig.add_hline(**hline)

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
                traces.extend([                # Bounds
                    go.Scatter(
                        name="Upper bound",
                        # legendgroup=label,
                        x=ts,
                        y=ub,
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    go.Scatter(
                        name="Lower bound",
                        # legendgroup=label,
                        x=ts,
                        y=lb,
                        line=dict(color='rgba(255,255,255,0)'),
                        fill="tonexty",
                        fillcolor=color_add_alpha(color, error_bars_alpha / sqrt(means.shape[0])),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                ])

            else:
                raise ValueError(f"Invalid mode: {mode}")

        fig.add_traces(traces)

        return fig

    plot_data = jt.leaves(
        tree_zip(vars_flat, means, stds, timesteps, labels),
        is_leaf=lambda x: isinstance(x, tuple),
    )

    for i, (var_flat, means, stds, ts, label) in enumerate(plot_data):
        fig = add_profile(
            fig,
            label,
            var_flat,
            means,
            means + n_std_plot * stds,
            means - n_std_plot * stds,
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


    fig.update_layout(legend_title_text=(
        "Condition" if legend_title is None else legend_title
    ))

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
        layout_modebar_add=['v1hovermode'],
        **kwargs,
    )

    if loss_context == "training":
        scatter_mode = "markers+lines"
    elif loss_context == "validation":
        # Validation losses are usually sparse in time, so don't connect with lines
        scatter_mode = "markers"
    else:
        raise ValueError(f"{loss_context} is not a valid loss context" )

    losses: LossDict | Array = jax.tree_map(
        lambda x: np.array(x),
        losses,
    )

    losses_total = losses if isinstance(losses, Array) else losses.total

    timesteps = pl.DataFrame({"timestep": range(losses_total.shape[0])})

    dfs = jax.tree_map(
        lambda losses: pl.DataFrame(losses),
        OrderedDict({"Total": losses_total}) | dict(losses),
    )

    # TODO: Only apply this when yaxis is log scaled
    dfs = jax.tree_map(
        lambda df: df.select([
            np.log10(pl.all()),
        ]),
        dfs,
    )

    loss_statistics = columns_mean_std(dfs)
    error_bars_bounds = errorbars(loss_statistics, n_std_plot)

    # TODO: Only apply this when yaxis is log scaled
    loss_statistics, error_bars_bounds = jax.tree_map(
        lambda df: timesteps.hstack(df.select([
            np.power(10, pl.col("*")),  # type: ignore
        ])),
        (loss_statistics, error_bars_bounds),
    )

    if colors is None:
        colors = DEFAULT_COLORS

    colors_rgb: list[str]
    colors_rgb, _ = convert_colors_to_same_type(colors, colortype='rgb')  # type: ignore

    colors_dict = {
        label: 'rgb(0,0,0)' if label == 'Total' else color
        for label, color in zip(dfs, colors_rgb)
    }

    for i, label in enumerate(reversed(dfs)):
        # Mean
        trace = go.Scatter(
            name=label,
            legendgroup=str(i),
            x=loss_statistics[label]["timestep"],
            y=loss_statistics[label]['mean'],
            mode=scatter_mode,
            marker_size=3,
            line=dict(color=colors_dict[label]),
        )
        trace.update(scatter_kws)
        fig.add_trace(trace)

        # Error bars
        fig.add_trace(go.Scatter(
            name="Upper bound",
            legendgroup=str(i),
            x=loss_statistics[label]["timestep"],
            y=error_bars_bounds[label]["ub"],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            name="Lower bound",
            legendgroup=str(i),
            x=loss_statistics[label]["timestep"],
            y=error_bars_bounds[label]["lb"],
            line=dict(color='rgba(255,255,255,0)'),
            fill="tonexty",
            fillcolor=color_add_alpha(colors_dict[label], error_bars_alpha),
            hoverinfo="skip",
            showlegend=False,
        ))

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
        fig.update_layout(hovermode='x')

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


def activity_heatmap(
    activity: Float[Array, "time unit"],
    colorscale: str = 'viridis',
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

    unit_idxs = jr.choice(
        key, np.arange(activities.shape[-1]), (n_samples,), replace=False
    )
    if unit_includes is not None:
        unit_idxs = np.concatenate([unit_idxs, np.array(unit_includes)])
    unit_idxs = np.sort(unit_idxs)
    unit_idx_strs = [str(i) for i in unit_idxs]

    xs = np.array(activities[..., unit_idxs])

    # Join all the data into a dataframe.
    df = pl.concat([
        # For each trial, construct all timesteps of x, y data.
        pl.from_numpy(x, schema=unit_idx_strs).hstack(pl.DataFrame({
            "Timestep": np.arange(x.shape[0]),
            # Note that "trial" here could be between- or within-condition.
            trial_label: pl.repeat(i, x.shape[0], eager=True),
        }))
        for i, x in enumerate(xs)
    ]).melt(
        id_vars=["Timestep", trial_label],
        value_name="Activity",
        variable_name="Unit",
    )

    fig = px.line(
        df,
        x='Timestep',
        y='Activity',
        color=trial_label,
        facet_row='Unit',
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
    fig.update_yaxes(zerolinewidth=0.5, zerolinecolor='black')

    # Improve formatting of subplot "Unit=" labels.
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.replace("=", " "),
        font=dict(size=14),
    ))

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


def tree_of_2d_timeseries_to_df(
    tree: PyTree[Shaped[Array, "condition timestep xy=2"], 'T'],
    labels: Optional[PyTree[str, 'T']] = None,
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
        return pl.concat([
            # For each trial, construct all timesteps of x, y data.
            pl.from_numpy(x, schema=["x", "y"]).hstack(pl.DataFrame({
                "Timestep": np.arange(x.shape[0]),
                "Condition": pl.repeat(i, x.shape[0], eager=True),
            }))
            for i, x in enumerate(array)
        ])

    xy_dfs = [get_xy_df(x) for x in arrays_flat]

    # Concatenate all variables.
    df = pl.concat([
        xy_df.hstack([pl.repeat(label, len(xy_df), eager=True).alias("var")])
        for xy_df, label in zip(xy_dfs, labels_flat)
    ])

    return df


def unshare_axes(fig: go.Figure):
    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)
    fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))


def effector_trajectories(
    states: SimpleFeedbackState | PyTree[Float[Array, "trial time ..."] | Any],
    where_data: Optional[
        Callable[[PyTree[Array]], tuple[Shaped[Array, "*batch trial time xy=2"]]]
    ] = None,
    var_labels: Optional[tuple[str, ...]] = None,
    step: int = 1,  # plot every step-th trial
    trial_specs: Optional["TaskTrialSpec"] = None,
    endpoints: Optional[
        tuple[Float[Array, "trial xy=2"], Float[Array, "trial xy=2"]]
    ] = None,
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

    if where_data is not None:
        vars_tuple = where_data(states)
        if var_labels is None:
            var_labels = where_func_to_attr_str_tree(where_data)

        if isinstance(vars_tuple, (Array, np.ndarray)):
            vars_tuple = (vars_tuple,)
            var_labels_ = (var_labels,)

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
            "If `states` is not a `SimpleFeedbackState`, "
            "`where_data` must be provided."
        )

    if len(vars_tuple[0].shape) > 3:
        # Collapse to a single batch dimension
        vars_tuple = jax.tree_map(
            lambda arr: np.reshape(arr, (-1, *arr.shape[-3:])),
            vars_tuple,
        )
        dfs = [
            tree_of_2d_timeseries_to_df(v, labels=var_labels_)
            for v in zip(*jax.tree_map(tuple, vars_tuple))
        ]
        dfs = [
            df.hstack(pl.DataFrame({"Trial": pl.repeat(i, len(df), eager=True)}))
            for i, df in enumerate(dfs)
        ]
        df = pl.concat(dfs, how='vertical')
    else:
        df = tree_of_2d_timeseries_to_df(
            vars_tuple,
            labels=var_labels_,
        )

    n_vars = df['var'].n_unique()

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
        colors = [str(c) for c in sample_colorscale('phase', n_conditions + 1)]

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color=df['Condition'].cast(pl.String),
        facet_col='var',
        facet_col_spacing=0.05,
        color_discrete_sequence=colors,
        labels=dict(color="Condition"),
        custom_data=['Condition', 'var', 'Timestep'],
        **kwargs,
    )

    if trace_kwargs is None:
        trace_kwargs = {}

    fig.for_each_trace(lambda trace: trace.update(
        mode=mode,
        hovertemplate=(
            'Condition: %{customdata[0]}<br>'
            'Time step: %{customdata[2]}<br>'
            'x: %{x:.2f}<br>'
            'y: %{y:.2f}<br>'
            '<extra></extra>'
        ),
        **trace_kwargs,
    ))

    fig.update_traces(marker_size=ms)

    if endpoints is not None:
        endpoints_arr = np.array(endpoints)  #type: ignore
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
                "Init": (ms_init, 'square'),
                "Goal": (ms_goal, 'circle'),
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
                            color='rgba(255, 255, 255, 0)',
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
                        line_dash='dash',
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
    fig.update_layout({
        f"yaxis{i}": dict(scaleanchor=f"x{i}")
        for i in [''] + list(range(2, n_vars + 1))
    })

    # Omit the "var=" part of each subplot title
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1],
        font=dict(size=14),
    ))

    # Facet plot shares axes by default.
    unshare_axes(fig)

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


def arr_to_nested_tuples(arr):
    """Like `ndarray.tolist()` but ensures the bottom-most level is tuples."""
    if arr.ndim == 1:
        return tuple(arr.tolist())
    elif arr.ndim > 1:
        return [arr_to_nested_tuples(sub_arr) for sub_arr in arr]
    else:
        return arr.item()  # For 0-dimensional arrays


def sample_colorscale_unique(colorscale, samplepoints: int, **kwargs):
    """Helper to ensure we don't get repeat colors when using cyclical colorscales.

    Also avoids the division-by-zero error that `sample_colorscale` raises when `samplepoints == 1`.
    """

    colors = plc.get_colorscale(colorscale)
    if samplepoints == 1:
        #! We could join this with the next condition, but I wanted to select the last rather than first
        # color in this case, for ad hoc reasons.
        n_sample = 2
        idxs = slice(1, None)
    elif colors[0][1] == colors[-1][1]:
        n_sample = samplepoints + 1
        idxs = slice(None, -1)
    else:
        n_sample = samplepoints
        idxs = slice(None)

    return sample_colorscale(colorscale, n_sample, **kwargs)[idxs]


def arr_to_rgb(arr):
    return f"rgb({', '.join(map(str, arr))})"


def adjust_color_brightness(colors, factor=0.8):
    colors_arr = np.array(plc.convert_colors_to_same_type(colors, colortype='tuple')[0])
    return list(map(arr_to_rgb, factor * colors_arr))


def trajectories_2D(
    # TODO: could just use tuple but then can't type T
    vars_: PyTree[Float[Array, "*trial time xy=2"], "T"],
    var_labels: Optional[PyTree[str, "T"]] = None,
    var_endpoint_ms: int = 0,
    show_mean: bool = True,
    lighten_mean: float = 0.8,
    # mode: Literal['curves', 'std'] = 'curves',
    # n_std: int = 1,
    colors: Optional[Float[Array, "*trial rgb=3"] | str] = None,
    colorscale_axis: int = 0,  # Can be any of the trial axes.
    colorscale: str = "phase",
    stride: int = 1,  # controls the stride over the `colorscale_axis` without needing to resize the input `vars_` and `legend_labels` etc
    n_curves_max: Optional[int] = None,
    legend_title: Optional[str] = None,
    legend_labels: Optional[Sequence] = None,
    curves_mode: str = "markers+lines",
    ms: int = 5,
    axes_labels: Optional[tuple[str, str] | PyTree[tuple[str, str], "T"]] = None,
    shared_yaxes_label: bool = True,
    layout_kws: Optional[dict] = None,
    scatter_kws: Optional[dict] = None,
    mean_scatter_kws: Optional[dict] = None,
    mean_exclude_axes: Sequence[int] = (),
    padding_factor: float = 0.1,
    **kwargs,
):
    """Plot 2D trajectories, with a subplot for each variable.

    This is a variant of `effector_trajectories` that should be easier to use for different trial/condition settings.

    Allows for multiple trial dimensions. Also allows us to colour the trajectories
    on a given trial dimension (e.g. color by trial vs. by reach direction).

    I imagine this function could be simplified even further by separating the PyTree/subplot logic from the trace generation
    logic. The trace generator just needs a set of array arguments, and returns a list of traces. These can then be
    added to one subplot or other. This could also mean separating (say) the endpoint traces off into a different function.

    Arguments:
        vars_: A PyTree of arrays, each of which define trajectories to be plotted on a separate subplot. Each array may have
            an arbitrary number of batch dimensions, all but one of which will be lumped together into groups of trajectories,
            by default.
        var_labels: The titles of the respective subplots.
        var_endpoint_ms: If non-zero, make the individual trajectory endpoints visible as markers of this size.
        show_mean: Show the average of the plotted trajectories for each trajectory group.
        lighten_mean: Factor by which to lighten (>1) or darken (<1) the color of the mean trajectory line.
        colors: Manually specify the colors. TODO.
        colorscale_axis: The axis in the arrays of `vars_` *not* to lump together, and which to color and display as groups
            on the legend.
        colorscale: The string name of the Plotly colorscale to use to color the elements of `colorscale_axis`.
        stride: The stride over the colorscale axis.
        n_curves_max: The maximum number of curves to plot for each lumped group; if the total exceeds this number, then
            we randomly sample this many from the group.
        mean_exclude_axes: Sequence of axes to exclude from the mean calculation when calculating mean trajectories.
            This allows you to preserve certain batch dimensions in the mean trajectories, which can be useful
            for showing multiple mean trajectories per color group.
    """
    # Assume all trajectory arrays have the same shape; i.e. matching trials between subplots
    # Add singleton dimensions if no trial dimensions are passed.
    vars_shapes = [v.shape for v in jt.leaves(vars_)]
    if not len(set(vars_shapes)) == 1:
        raise ValueError("All trajectory arrays should have the same shape")

    if len(vars_shapes[0]) < 2:
        raise ValueError("Trajectory arrays must have at least 2 axes (time and space)!")

    if len(vars_shapes[0]) == 2:
        vars_ = jt.map(lambda v: v[None, :], vars_)

    if stride != 1:
        n_colors = jt.leaves(vars_)[0].shape[colorscale_axis]
        idxs = np.arange(n_colors)[::stride]
        vars_ = tree_take(vars_, idxs, colorscale_axis)

    vars_shape = jt.leaves(vars_)[0].shape
    n_curves = np.prod(vars_shape[:-2]) // vars_shape[colorscale_axis]
    n_trial_axes = len(vars_shape) - 2

    if var_labels is None:
        var_labels = tree_labels(vars_)

    if legend_title is None:
        legend_title = "Condition"

    var_labels = tree_prefix_expand(var_labels, vars_, is_leaf=is_none)

    if scatter_kws is None:
        scatter_kws = {}

    if colorscale_axis < 0:
        colorscale_axis += len(vars_shape)

    # Convert colorscale_axis to positive index if negative
    full_ndim = len(vars_shape)
    colorscale_axis_pos = colorscale_axis if colorscale_axis >= 0 else full_ndim + colorscale_axis

    # Ensure colorscale_axis is within batch dimensions (not time or xy)
    if colorscale_axis_pos >= full_ndim - 2:
        raise ValueError(f"colorscale_axis {colorscale_axis} points to a non-batch dimension")

    constant_color_batch_axes = tuple(i for i in range(len(vars_shape[:-2])) if i != colorscale_axis_pos)

    # Determine the RGB colours of every point, before plotting.
    if colors is None:
        color_sequence = np.array(
            sample_colorscale_unique(colorscale, vars_shape[colorscale_axis], colortype='tuple')
        )
        colors, color_idxs = jt.map(
            lambda x, finalshape: np.broadcast_to(
                np.expand_dims(x, constant_color_batch_axes),
                vars_shape[:-2] + finalshape,
            ),
            (color_sequence, np.arange(vars_shape[colorscale_axis])), ((3,), ()),
        )
        # var_colors, var_color_idxs = jt.map(
        #     lambda x: jt.map(lambda _: x, vars_),
        #     (colors, color_idxs),
        # )
    # elif isinstance(colors, str):
    #     colors = np.full(vars_shape[:-2], str(colors))
    #     color_idxs = None
    else:
        color_sequence = ()
        color_idxs = None

    # TODO: Don't plot mean trajectories if user manually specifies colors?
    # TODO: Alternatively, get rid of manual specification of colors
    if show_mean: # or mode == "std":
        # Convert negative indices in mean_exclude_axes to positive
        full_ndim = len(vars_shape)
        mean_exclude_axes_pos = tuple(ax if ax >= 0 else full_ndim + ax for ax in mean_exclude_axes)

        # Calculate axes to use for the mean calculation
        mean_axes = tuple(ax for ax in constant_color_batch_axes if ax not in mean_exclude_axes_pos)
        mean_vars = jt.map(
            lambda x: np.nanmean(x, axis=mean_axes),
            vars_,
        )

        # Calculate color indices for mean vars - will do this after we have constructed color_idxs
        # We'll save this here for now
        mean_color_idxs = None

        # If mean_exclude_axes is provided, we need to reshape mean_vars
        if mean_exclude_axes_pos:
            # Adjust colorscale_axis position after excluding axes
            adjusted_colorscale_axis = colorscale_axis
            for ax in sorted(mean_exclude_axes_pos):
                if ax < colorscale_axis:
                    adjusted_colorscale_axis -= 1

            # Now reshape mean_vars similar to vars_, keeping colorscale axis as is
            mean_vars = jt.map(
                lambda x: np.reshape(
                    np.moveaxis(x, adjusted_colorscale_axis, 0),
                    (x.shape[adjusted_colorscale_axis], -1, *x.shape[-2:])
                ),
                mean_vars,
            )
        else:
            mean_vars = jt.map(
                lambda x: np.reshape(x, (x.shape[0], 1, *x.shape[1:])),
                mean_vars,
            )
    else:
        mean_vars = {}

    # Now that we have color_idxs, compute mean_color_idxs
    if show_mean and mean_exclude_axes_pos and color_idxs is not None:
        # Take one value from each of the mean axes
        # (They all have same color index by construction)
        slices = [slice(None)] * len(vars_shape[:-2])
        for ax in mean_axes:
            slices[ax] = 0

        # Get pre-flattened color indices with mean axes removed
        mean_color_indices_pre = color_idxs[tuple(slices)]

        # Now reshape mean_color_indices_pre in the same way as mean_vars
        mean_color_idxs = np.reshape(
            np.moveaxis(mean_color_indices_pre, adjusted_colorscale_axis, 0),
            (mean_color_indices_pre.shape[adjusted_colorscale_axis], -1)
        )

    # Collapse all trial axes but `colorscale_axis`, and subsample if necessary
    vars_, colors, color_idxs = jt.map(
        lambda x: np.reshape(
            np.moveaxis(x, colorscale_axis, 0),
            (vars_shape[colorscale_axis], -1, *x.shape[n_trial_axes:])
        ),
        (vars_, colors, color_idxs),
    )

    if n_curves_max is not None and n_curves > n_curves_max:
        idxs_sample = np.random.choice(np.arange(n_curves), n_curves_max, replace=False).astype(int)
        vars_, colors, color_idxs = jt.map(
            lambda x: x[:, idxs_sample],
            (vars_, colors, color_idxs),
        )

    # Collapse into a single trial dimension.
    vars_, colors, color_idxs = jt.map(
        lambda x: np.reshape(x, (-1, *x.shape[2:])),
        (vars_, colors, color_idxs),
    )

    colors_rgb_tuples = arr_to_nested_tuples(colors)
    colors_rgb = jt.map(
        lambda x: convert_colors_to_same_type(x)[0][0],
        colors_rgb_tuples,
        is_leaf=lambda x: isinstance(x, tuple),
    )
    color_idxs = color_idxs.tolist()
    if legend_labels is None:
        legend_labels = np.arange(vars_shape[colorscale_axis]).tolist()

    assert len(legend_labels[::stride]) == vars_shape[colorscale_axis], (
        "Number of legend labels should match size of the `colorscale_axis`"
    )

    ts = np.arange(jt.leaves(vars_)[0].shape[-2])

    subplots_data = jt.leaves(
        tree_zip(vars_, var_labels),
        is_leaf=lambda x: isinstance(x, tuple) and isinstance(x[0], Array),
    )

    # TODO: could instead tree_map over `var_labels` (as prefix) to get subtrees of data
    n_subplots = len(subplots_data)

    subplot_titles = [f"<b>{label}</b>" for label in jt.leaves(var_labels)]

    fig = make_subplots(
        rows=1, cols=n_subplots,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.1,
    )

    if shared_yaxes_label:
        for col in range(1, n_subplots + 1):
            fig.update_yaxes(title_text="", row=1, col=col)

    # Constrain the axes of each subplot to be scaled equally.
    # (i.e. square aspect ratio)
    fig.update_layout({
        f"yaxis{i}": dict(scaleanchor=f"x{i}")
        for i in [''] + list(range(2, n_subplots + 1))
    })

    # Track whether each color has been added to the legend yet during plotting of trial trajectories;
    # if mean trajectories are enabled then those are the ones we'll add (later).
    color_added_to_legend = dict.fromkeys(color_idxs, False)

    def plot_var(fig, col, var, label):
        assert len(var.shape) < 4, "The data shouldn't have this many axes; we reshaped it!"

        traces = []

        for i, xy in enumerate(var):
            # Only add each color to the legend the first time,
            # and only if we're not plotting mean trajectories.
            showlegend = False
            if not show_mean and not color_added_to_legend[color_idxs[i]]:
                showlegend = True
                color_added_to_legend[color_idxs[i]] = True

            trace = go.Scatter(
                name=legend_labels[::stride][color_idxs[i]],
                showlegend=showlegend,
                # legendgroup=color_idxs[i],  # controls show/hide with other traces
                x=xy[..., 0],
                y=xy[..., 1],
                mode=curves_mode,
                marker_size=ms,
                line=dict(
                    color=colors_rgb[i]
                ),
                customdata=np.concatenate(
                    [
                        ts[:, None],
                        np.broadcast_to([[i, legend_labels[::stride][color_idxs[i]]]], (ts.shape[0], 2))
                    ],
                    axis=-1,
                ),
                **scatter_kws,
            )
            traces.append(trace)

            if var_endpoint_ms > 0:
                trace = go.Scatter(
                    name=f"{legend_labels[::stride][color_idxs[i]]} endpoint",
                    showlegend=False,
                    # legendgroup=color_idxs[i],  # controls show/hide with other traces
                    x=xy[..., -1, 0][None],
                    y=xy[..., -1, 1][None],
                    mode="markers",
                    marker_size=var_endpoint_ms,
                    line=dict(
                        color=colors_rgb[i]
                    ),
                    marker_symbol="circle-open",
                    # **scatter_kws,
                )
                traces.append(trace)

        for trace in traces:
            fig.add_trace(trace, row=1, col=col)

        return fig

    all_vars_flat = jt.leaves(vars_, is_leaf=eqx.is_array)

    # if mode == "curves":
    for i, (var_data, subplot_label) in enumerate(zip(all_vars_flat, subplot_titles)):
        col = i + 1
        fig = plot_var(fig, col, var_data, subplot_label)

        all_y_data = var_data[..., 1].flatten()

        # Check if there are any valid y-data points before calculating range
        if not np.all(np.isnan(all_y_data)):
            y_min = np.nanmin(all_y_data)
            y_max = np.nanmax(all_y_data)
            y_range = y_max - y_min
            # Handle case where range is zero (e.g., constant y)
            padding = y_range * padding_factor if y_range > 0 else 0.1 # Add small absolute padding if range is 0

            # --- Update Y-axis range ---
            fig.update_yaxes(range=[y_min - padding, y_max + padding], row=1, col=col)

            fig.update_xaxes(autorange=True, row=1, col=col)

    if show_mean:
        mean_kwargs = dict(
            line_width=2,
            opacity=1,
        )

        if mean_scatter_kws is not None:
            mean_kwargs |= mean_scatter_kws

        def _plot_mean_var(i, mean_var):
            # Loop over the axis=0 color groups
            for j, curves in enumerate(mean_var):
                # Loop over axis=1 curves within this color group (due to mean_exclude_axes)
                for k, xy in enumerate(curves):
                    # Get the true color index for this curve
                    if mean_color_idxs is not None:
                        color_idx = int(mean_color_idxs[j][k])
                    else:
                        color_idx = j

                    trace = go.Scatter(
                        name=legend_labels[::stride][color_idx],
                        # Only show in legend once per group
                        showlegend=(i==0),
                        # legendgroup=color_idx,  # controls show/hide with other traces
                        x=xy[..., 0],
                        y=xy[..., 1],
                        mode="lines",
                        line_color=arr_to_rgb(lighten_mean * color_sequence[color_idx]),
                        customdata=np.concatenate(
                            [
                                ts[:, None],
                                np.broadcast_to([[color_idx, legend_labels[::stride][color_idx]]], (ts.shape[0], 2)),
                            ],
                            axis=-1,
                        ),
                        **mean_kwargs,
                    )
                    fig.add_trace(trace, row=1, col=i + 1)

        jt.map(_plot_mean_var, *zip(*enumerate(jt.leaves(mean_vars))))

    fig.for_each_trace(lambda trace: trace.update(
        hovertemplate=(
            f'{legend_title}: %{{customdata[2]}}<br>'
            'Time step: %{customdata[0]}<br>'
            'x: %{x:.2f}<br>'
            'y: %{y:.2f}<br>'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(legend_itemsizing="constant")
    fig.update_layout(legend_title_text=legend_title)

    if axes_labels is not None:
        fig.update_xaxes(title_text=axes_labels[0])
        fig.update_yaxes(title_text=axes_labels[1])

    if layout_kws is not None:
        fig.update_layout(layout_kws)

    return fig


def is_trace(element):
    return isinstance(element, BaseTraceType)


def trajectories_3D(
    traj: Float[Array, "trials time 3"],
    endpoint_symbol: Optional[str] = 'circle-open',
    start_symbol: Optional[str] = None,
    fig: Optional[go.Figure] = None,
    colors: str | Sequence[str | None] | None = None,
    axis_labels: Optional[tuple[str, str, str]] = None,
    mode: str = 'lines',
    name: Optional[str] = "State trajectory",
    **kwargs,
):
    """Plot 3D trajectories."""
    if fig is None:
        fig = go.Figure(layout=dict(width=1000, height=1000))

    if colors is None or isinstance(colors, str):
        colors_func = lambda _: colors
    else:
        colors_func = lambda i: colors[i]

    idxs = np.arange(traj.shape[1])

    if axis_labels is None:
        axis_labels = ('x', 'y', 'z')
    else:
        fig.update_layout(
            scene=dict(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2],
            )
        )

    if start_symbol is not None:
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0, 0],
                y=traj[:, 0, 1],
                z=traj[:, 0, 2],
                mode='markers',
                marker_symbol=start_symbol,
                marker_line_width=2,
                marker_size=10,
                marker_color="None",
                marker_line_color=colors,
                name=f'{name} start'
            )
        )
    fig.add_traces(
        [
            go.Scatter3d(
                x=traj[idx, :, 0],
                y=traj[idx, :, 1],
                z=traj[idx, :, 2],
                mode=mode,
                line_color=colors_func(idx),
                marker_color=colors_func(idx),
                # name='Reach trajectories',
                showlegend=(name is not None and idx == 0),
                name=name,
                customdata=idxs[:, None],
                hovertemplate=(
                    'index: %{customdata[0]}<br>'
                    f'{axis_labels[0]}: %{{x:.2f}}<br>'
                    f'{axis_labels[1]}: %{{y:.2f}}<br>'
                    f'{axis_labels[2]}: %{{z:.2f}}<br>'
                    '<extra></extra>'
                ),
                legendgroup=name,
                **kwargs,
            )
            for idx in range(traj.shape[0])
        ]
    )
    if endpoint_symbol is not None:
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, -1, 0],
                y=traj[:, -1, 1],
                z=traj[:, -1, 2],
                mode='markers',
                marker_symbol=endpoint_symbol,
                marker_line_width=2,
                marker_size=5,
                marker_color=colors,
                marker_line_color=colors,
                name=f'{name} end'
            )
        )
    fig.update_layout(
        legend_tracegroupgap=1,
    )

    return fig


def plot_eigvals(
    eigvals: Float[Array, "batch eigvals"],
    colors: str | Sequence[str] | None = None,
    colorscale: str = 'phase',
    labels: Optional[Sequence[Optional[str]]] = None,
    mode: str = 'markers',
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
        type='circle',
        xref='x', yref='y',
        x0=-1, y0=-1, x1=1, y1=1,
        line_color='black',
    )

    # Add dashed axes lines
    fig.add_hline(0, line_dash='dot', line_color='grey')
    fig.add_vline(0, line_dash='dot', line_color='grey')

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


