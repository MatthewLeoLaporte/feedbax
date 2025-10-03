import numpy as np
import plotly.colors as plc
import plotly.io as pio
from plotly.colors import convert_colors_to_same_type, sample_colorscale

DEFAULT_COLORS = pio.templates[pio.templates.default].layout.colorway  # pyright: ignore


def color_add_alpha(rgb_str: str, alpha: float):
    return f"rgba{rgb_str[3:-1]}, {alpha})"


def arr_to_rgb(arr):
    return f"rgb({', '.join(map(str, arr))})"


def adjust_color_brightness(colors, factor=0.8):
    colors_arr = np.array(plc.convert_colors_to_same_type(colors, colortype="tuple")[0])
    return list(map(arr_to_rgb, factor * colors_arr))


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
