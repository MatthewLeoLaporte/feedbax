"""Base classes for continuous dynamical systems.

:copyright: Copyright 2023-2024 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
import logging 
from typing import Generic, Optional, Protocol, TypeVar

import equinox as eqx
from equinox import AbstractVar
import jax 
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax.state import AbstractState, CartesianState2D, StateBounds


logger = logging.getLogger(__name__)

    
StateT = TypeVar("StateT", bound=AbstractState)


class AbstractDynamicalSystem(eqx.Module, Generic[StateT]):
    """Base class for continuous dynamical systems.
    
    This is a module that provides a vector field for use as
    with `diffrax.ODETerm`.
    """
    
    @abstractmethod
    def __call__(
        self, 
        t: float, 
        state: StateT, 
        input: PyTree[Array],  # controls
    ) -> StateT:
        """Vector field of the system."""
        ...

    @abstractproperty
    def control_size(self) -> int:
        """Number of control inputs."""
        ...
    
    @abstractmethod
    def init(self, *, key: Optional[PRNGKeyArray] = None) -> StateT:
        """Returns the initial state of the system.
        """
        ...
    
    @abstractproperty
    def bounds(self) -> StateBounds[StateT]:
        """Suggested bounds on the state.
        
        These will only be applied if...
        """
        ...
        

class AbstractLTISystem(AbstractDynamicalSystem[CartesianState2D]):
    """A linear, continuous, time-invariant system.
    
    Inspired by https://docs.kidger.site/diffrax/examples/kalman_filter/
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
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> CartesianState2D:
        return CartesianState2D()