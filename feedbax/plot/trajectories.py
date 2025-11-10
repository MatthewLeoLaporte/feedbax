from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeGuard, ValuesView

import jax.tree as jt
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import polars as pl
from jaxtyping import Array, Float, PyTree, Shaped
from plotly.basedatatypes import BaseTraceType
from plotly.colors import convert_colors_to_same_type, sample_colorscale
from plotly.subplots import make_subplots

from feedbax import tree_labels
from feedbax.bodies import SimpleFeedbackState
from feedbax.misc import where_func_to_attr_str_tree
from feedbax.plot.colors import _compute_colors, arr_to_rgb
from feedbax.plot.misc import (
    AxesLabels,
    StrMapping,
    _is_str_mapping,
    _mean_over_axes,
    _normalize_vars,
    arr_to_nested_tuples,
    tree_of_2d_timeseries_to_df,
    unshare_axes,
)
from feedbax.types import SeqOf, SeqOfT
from jax_cookbook import MaskedArray

if TYPE_CHECKING:
    from feedbax.task import TaskTrialSpec


def _unwrap_masked_array(x):
    """Convert MaskedArray to regular array with masked values set to NaN."""
    if isinstance(x, MaskedArray):
        return np.asarray(x.unwrap())
    return x


def effector_trajectories(
    states: SimpleFeedbackState | PyTree[Float[Array, "trial time ..."] | Any],
    where_data: Optional[
        Callable[[PyTree[Array]], Sequence[Shaped[Array, "*batch trial time xy=2"]]]
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


TRAJ_SCATTER_KWS_DEFAULT: StrMapping = MappingProxyType(
    dict(
        mode="lines",
        marker_size=5,
    )
)


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

    Supports MaskedArray leaves (duck-typed via .data/.mask attributes), which are
    converted to NaN-masked arrays.
    """
    # Unwrap any MaskedArray instances to NaN-masked regular arrays
    subplots_data = jt.map(_unwrap_masked_array, subplots_data)

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
