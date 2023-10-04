""" 

:copyright: Copyright 2023 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

import logging 

import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float
import numpy as np


logger = logging.getLogger(__name__)


def uniform_endpoints(
    key: jrandom.PRNGKey,
    batch_size: int, 
    ndim: int = 2, 
    workspace: Float[Array, "ndim 2"] = jnp.array([[-1., 1.], 
                                                   [-1., 1.]]),
):
    """Segment endpoints uniformly distributed in a rectangular workspace."""
    return jrandom.uniform(
        key, 
        (2, batch_size, ndim),   # (start/end, ...)
        minval=workspace[:, 0], 
        maxval=workspace[:, 1]
    )


def centreout_endpoints(
    center: Float[Array, "2"], 
    n_directions: int, 
    angle_offset: float, 
    length: float,
): 
    ndim = 2  # TODO: generalize to sphere?
    """Segment endpoints starting in the centre and ending equally spaced on a circle."""
    angles = jnp.linspace(0, 2 * np.pi, n_directions + 1)[:-1]
    angles = angles + angle_offset

    starts = jnp.tile(center, (n_directions, 1))
    ends = center + length * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    return jnp.stack([starts, ends], axis=0)  