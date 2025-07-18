"""Base classes for continuous dynamical systems.

:copyright: Copyright 2023-2024 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
import logging
from typing import Optional, TypeVar

from equinox import Module
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar

from feedbax._model import AbstractModel
from feedbax.state import StateBounds, StateT


logger = logging.getLogger(__name__)


# StateT = TypeVar("StateT", Module, Array)


class AbstractDynamicalSystem(AbstractModel[StateT]):
    """Base class for continuous dynamical systems.

    ??? dev-note "Development note"
        The signature of `vector_field` matches that expected by
        `diffrax.ODETerm`. However, the argument that is called
        `args` in the Diffrax documentation, we call `input`, since
        we use it for input signals (e.g. forces from the controller)

        Vector fields for biomechanical models are generally not
        time-dependent. That is, the argument `t` to `vector_field` is
        typically unused. This is apparent in the way we alias `vector_field`
        to `__call__`, which is a method that `AbstractModel` requires.

        Perhaps it is unnecessary to inherit from `AbstractModel`, though.
    """

    def __call__(
        self,
        input_: PyTree[Array],
        state: StateT,
        key: PRNGKeyArray,
    ) -> StateT:
        """Alias for `vector_field`, with a modified signature."""
        return self.vector_field(jnp.array(0.), state, input_)

    @abstractmethod
    def vector_field(
        self,
        t: Scalar,
        state: StateT,
        input: PyTree[Array],  # controls
    ) -> StateT:
        """Returns scalar (e.g. time) derivatives of the system's states."""
        ...

    @abstractproperty
    def input_size(self) -> int:
        """Number of input variables."""
        ...

    @abstractmethod
    def init(self, *, key: PRNGKeyArray) -> StateT:
        """Returns the initial state of the system."""
        ...

    @property
    def step(self) -> Module:
        return self


class LTISystem(AbstractDynamicalSystem[Array]):
    """
    !!! ref inline end ""
        Inspired by [this Diffrax example](https://docs.kidger.site/diffrax/examples/kalman_filter/).

    A linear, continuous, time-invariant system.

    Attributes:
        A: The state evolution matrix.
        B: The control matrix.
        C: The observation matrix.
    """

    A: Float[Array, "state state"]
    B: Float[Array, "state input"]
    C: Float[Array, "obs state"]

    @jax.named_scope("fbx.AbstractLTISystem")
    def vector_field(
        self,
        t: Scalar,
        state: Array,
        input: Array,
    ) -> Array:
        """Returns time derivatives of the system's states."""
        return self.A @ state + self.B @ input

    @property
    def input_size(self) -> int:
        """Number of control variables."""
        return self.B.shape[1]

    @property
    def state_size(self) -> int:
        """Number of state variables."""
        return self.A.shape[1]

    @property
    def bounds(self) -> StateBounds[Array]:
        return StateBounds(low=None, high=None)

    def init(
        self,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Array:
        """Return a default state for the linear system."""
        return jnp.zeros(self.state_size)
