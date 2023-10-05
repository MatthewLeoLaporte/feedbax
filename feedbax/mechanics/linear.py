"""

:copyright: Copyright 2023 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

import logging 
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


logger = logging.getLogger(__name__)


# global defaults
ORDER = 2  # maximum order of the state derivatives
N_DIM = 2  # number of spatial dimensions


class LTISystem(eqx.Module):
    """Linear time-invariant system.
    
    Note that the system is defined in continuous time.
    
    Inspired by https://docs.kidger.site/diffrax/examples/kalman_filter/
    """
    A: Float[Array, "state state"]  # state evolution matrix
    B: Float[Array, "state input"]  # control matrix
    C: Float[Array, "obs state"]  # observation matrix
    
    def vector_field(
        self, 
        t: float, 
        y: Float[Array, "state"],
        args: Float[Array, "input"]
    ) -> Float[Array, "state"]:
        u = args  
        d_y = self.A @ jnp.concatenate(y) + self.B @ u
        # TODO: don't hardcode the split; define on instantiation
        d_pos, d_vel = d_y[:2], d_y[2:]
        return d_pos, d_vel
    
    @property
    def control_size(self) -> int:
        return self.B.shape[1]
    
    @property 
    def state_size(self) -> int:
        return self.A.shape[1]
    

def point_mass(
    mass: float = 1., 
    order: int = ORDER, 
    n_dim: int = N_DIM
) -> LTISystem:
    A = sum([jnp.diagflat(jnp.ones((order - i) * n_dim), 
                          i * n_dim)
             for i in range(1, order)])
    B = jnp.concatenate(
        [jnp.zeros((n_dim, n_dim)), 
         jnp.eye(n_dim) / mass], 
        axis=0
    )
    C = jnp.array(1)  # TODO 
    return LTISystem(A, B, C)