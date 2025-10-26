"""

:copyright: Copyright 2024 by Matt L. Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from collections import OrderedDict
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    TypeVar,
)

import jax.random as jr
import jax.tree as jt
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import polars as pl
from jax_cookbook import identity
from jaxtyping import Array, Float, PRNGKeyArray

# pyright: reportMissingTypeStubs=false
from plotly.colors import convert_colors_to_same_type

from feedbax.loss import AbstractLoss, TermTree
from feedbax.plot.colors import DEFAULT_COLORS, color_add_alpha
from feedbax.plot.misc import columns_mean_std, errorbars

logger = logging.getLogger(__name__)


T = TypeVar("T")


def loss_history(
    losses: TermTree | Array,
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

    losses = jt.map(np.array, losses)

    losses_total = losses if isinstance(losses, Array) else losses.aggregate(leaf_fn=identity)

    training_iterations = pl.DataFrame({"iteration": range(losses_total.shape[0])})

    dfs = jt.map(
        lambda losses: pl.DataFrame(np.array(losses)),
        OrderedDict({"Total": losses_total}) | losses.flatten(),
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
        lambda df: training_iterations.hstack(
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
            x=loss_statistics[label]["iteration"],
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
                x=loss_statistics[label]["iteration"],
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
                x=loss_statistics[label]["iteration"],
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


# Backwards compatible wrappers
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
