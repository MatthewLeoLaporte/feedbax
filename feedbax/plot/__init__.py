import plotly.io as pio

from .colors import (
    adjust_color_brightness,
    sample_colorscale_unique,
)
from .misc import AxesLabels
from .plotly import loss_history
from .profiles import profiles
from .trajectories import (
    trajectories,
    trajectories_2D,
    trajectories_3D,
)

pio.templates.default = "plotly_white"
