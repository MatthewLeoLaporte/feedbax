"""Base classes for continuous dynamical systems.

:copyright: Copyright 2023 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

import logging 
from typing import Generic, Protocol, TypeVar

import equinox as eqx
from equinox import AbstractVar
import jax 
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from feedbax.state import AbstractState, CartesianState2D, StateBounds


logger = logging.getLogger(__name__)

    
StateT = TypeVar("StateT", bound=AbstractState)


class AbstractDynamicalSystem(eqx.Module, Generic[StateT]):
    """Base class for continuous dynamical systems
    
    TODO:
    - Signature of `init`. Versus the Diffrax documentation, I'm using `state` 
      for `y` and `input` for `args`. Unfortunately this is reversed from the 
      signature of `AbstractModel`. Perhaps we could automatically wrap all 
      subclasses of `AbstractDynamicalSystem` to make this consistent.
    - All of the biomechanical models (so far) are time invariant, i.e. they 
      don't vary with time itself. Perhaps exclude it from the signature as 
      well.
    """
    
    def __call__(
        self, 
        t: float, 
        state: StateT, 
        input: PyTree,  # controls
    ) -> StateT:
        """Vector field of the system."""
        ...

    @property
    def control_size(self) -> int:
        """Number of control inputs."""
        ...
    
    def init(self, **kwargs) -> StateT:
        """Initial state of the system.
        
        TODO:
        
        - This is shared with `AbstractModel`, so perhaps they should have 
          a common abstract parent.
        """
        ...
    
    @property
    def bounds(self) -> StateBounds[StateT]:
        """Suggested bounds on the state.
        
        These will only be applied if...
        """
        ...
        

class AbstractLTISystem(AbstractDynamicalSystem[CartesianState2D]):
    """Linear time-invariant system.
    
    Note that the system is defined in continuous time.
    
    Inspired by https://docs.kidger.site/diffrax/examples/kalman_filter/
    
    TODO:
    - Don't hardcode the state split.
    - Should it be CartesianState2D... linear non-Cartesian systems?
    """
    A: AbstractVar[Float[Array, "state state"]]  # state evolution matrix
    B: AbstractVar[Float[Array, "state input"]]  # control matrix
    C: AbstractVar[Float[Array, "obs state"]]  # observation matrix
    
    @jax.named_scope("fbx.AbstractLTISystem")
    def __call__(
        self, 
        t: float, 
        state: StateT,
        input: Float[Array, "input"]
    ) -> Float[Array, "state"]:
        force = input + state.force
        state_ = jnp.concatenate([state.pos, state.vel])       
        d_y = self.A @ state_ + self.B @ force
        d_pos, d_vel = d_y[:2], d_y[2:]
        
        return CartesianState2D(pos=d_pos, vel=d_vel)
    @property
    def control_size(self) -> int:
        return self.B.shape[1]
    
    @property 
    def state_size(self) -> int:
        return self.A.shape[1]
    
    @property
    def bounds(self) -> StateBounds[CartesianState2D]:
        return StateBounds(low=None, high=None)
    
    def init(
        self,
    ) -> CartesianState2D:
        return CartesianState2D()