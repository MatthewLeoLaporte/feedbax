"""General-purpose filters for dynamical systems.

:copyright: Copyright 2023-2024 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

from functools import cached_property
from typing import Optional, Type

import diffrax as dfx
import equinox as eqx
from equinox import field
import jax
import jax.tree as jt
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from feedbax.dynamics import AbstractDynamicalSystem
from feedbax._model import AbstractModel
from feedbax.state import StateBounds


class FilterState(eqx.Module):
    """Holds the current filtered signal and solver state.

    Attributes:
        output: The current filtered signal.
        solver: The state of the Diffrax solver.
    """
    output: jax.Array  # shape = (*, n_dof)
    solver: PyTree


class FirstOrderFilter(AbstractDynamicalSystem[FilterState]):
    """
    Continuous-time first-order low-pass with optional asymmetric
    rise / decay time-constants:

        ẋ = (u − x) / τ_rise    if u ≥ x
          = (u − x) / τ_decay   if u <  x
    """
    tau_rise: float = 0.050   # seconds
    tau_decay: float = 0.050  # seconds (set > tau_rise for slower relaxation)
    dt: float = 0.001        # time step duration
    solver: dfx.AbstractSolver = field(default_factory=lambda: dfx.Euler())
    input_proto: PyTree[Array] = field(default_factory=lambda: jnp.zeros(1))
    init_value: float = 0.0

    def vector_field(
        self,
        t: Scalar,                    # simulation time
        state: FilterState,          # current output state
        input: PyTree[jax.Array],           # command (same shape as output)
    ) -> FilterState:
        """Return the time derivative of the filtered signal."""
        tau = jnp.where(input >= state.output, self.tau_rise, self.tau_decay)
        return eqx.tree_at(
            lambda state: state.output,
            state,
            (input - state.output) / tau
        )

    @cached_property
    def _term(self) -> dfx.AbstractTerm:
        """The Diffrax term for the filter dynamics."""
        return dfx.ODETerm(self.vector_field)

    def __call__(
        self,
        input_: PyTree[Array],
        state: FilterState,
        key: PRNGKeyArray,
    ) -> FilterState:
        """Return an updated state after a single step of filter dynamics."""
        output_state, _, _, solver_state, _ = self.solver.step(
            self._term,
            0,
            self.dt,
            state,
            input_.value,
            state.solver,
            made_jump=False,
        )

        return eqx.tree_at(
            lambda state: state.solver,
            output_state,
            solver_state,
        )

    @property
    def memory_spec(self):
        """Tell the staging system which field is stateful."""
        return FilterState(output=True, solver=False)

    def init(self, *, key: Optional[PRNGKeyArray] = None):
        """Returns an initial FilterState based on input_proto and init_value."""
        output_init = jt.map(
            lambda x: jnp.full_like(x, self.init_value), self.input_proto
        )
        solver_init = self.solver.init(self._term, 0, self.dt, output_init, None)
        return FilterState(output=output_init, solver=solver_init)

    def change_input(self, input_proto: PyTree[Array]) -> "FirstOrderFilter":
        """Returns a similar FirstOrderFilter with a changed input structure."""
        return eqx.tree_at(lambda filter: filter.input_proto, self, input_proto)

    @property
    def input_size(self) -> int:
        """Number of input variables (inferred from example during init)."""
        # This is abstract but we can't determine size without an example
        raise NotImplementedError("Input size depends on the shape provided during init")

    @property
    def bounds(self) -> StateBounds[FilterState]:
        """Specifies the bounds of the filter state."""
        return StateBounds(
            low=FilterState(output=None, solver=None),
            high=FilterState(output=None, solver=None)
        )