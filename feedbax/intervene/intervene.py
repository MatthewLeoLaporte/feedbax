"""Add-ins to a model that modify its state operations.

Intervenors are intended to be used with instances of types of `AbstractStagedModel`, to
patch state operations onto existing models. For example, it is common to take
an existing task and apply a certain force field on the biomechanical effector.
Instead of rewriting the biomechanical model itself, which would lead to a
proliferation of model classes with slightly different task conditions, we can
instead use `add_fixed_intervenor` to do surgery on the model using only a few lines
of code.

Likewise, since the exact parameters of an intervention often change between
(or within) task trials, it is convenient to be able to specify the distribution
of these parameters across trials. The function `schedule_intervenors` does
simultaneous surgery on task and model instances so that the model interventions
are paired with their trial-by-trial parameters, as provided by the task.

TODO:
- Could just use `operation` to distinguish `NetworkClamp` from
  `NetworkConstantInput`. Though might need to rethink the NaN unit spec thing
- Could enforce that `out_where = in_where` for some intervenors, in particular
  `AddNoise`, `NetworkClamp`, and `NetworkConstantInput`. These are intended to
  make modifications to a part of the state, not to transform between parts.

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

import copy
import logging
import operator as op
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import fields
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    Self,
    TypeVar,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax.tree_util as jtu
from equinox import AbstractVar, Module, field
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray, PyTree

from feedbax.noise import Normal
from feedbax.state import StateT

if TYPE_CHECKING:
    # from feedbax.model import AbstractModel
    from feedbax.mechanics.mechanics import MechanicsState
    from feedbax.nn import NetworkState


logger = logging.getLogger(__name__)


class AbstractIntervenorInput(Module):
    """Base class for PyTrees of intervention parameters.

    Attributes:
        active: Whether the intervention is active.
        scale: Factor by which the intervenor output is scaled.
    """

    active: AbstractVar[bool]
    scale: AbstractVar[float]


InputT = TypeVar("InputT", bound=AbstractIntervenorInput)


class AbstractIntervenor(Module, Generic[StateT, InputT]):
    """Base class for modules that intervene on a model's state.

    Attributes:
        params: Default intervention parameters.
        in_where: Takes an instance of the model state, and returns the substate
            corresponding to the intervenor's input.
        out_where: Takes an instance of the model state, and returns the substate
            corresponding to the intervenor's output. In many cases, `out_where` will
            be the same as `in_where`.
        operation: Which operation to use to combine the original and altered
            `out_where` substates. For example, an intervenor that clamps a state
            variable to a particular value should use an operation like `lambda x, y: y`
            to replace the original with the altered state. On the other hand, an
            additive intervenor would use the equivalent of `lambda x, y: x + y`.
    """

    params: AbstractVar[InputT]
    in_where: AbstractVar[Callable[[StateT], PyTree[ArrayLike, " T"]]]
    out_where: AbstractVar[Callable[[StateT], PyTree[ArrayLike, " S"]]]
    operation: AbstractVar[Callable[[ArrayLike, ArrayLike], ArrayLike]]

    @classmethod
    def with_params(cls, **kwargs) -> Self:
        """Constructor that accepts field names of `InputT` as keywords.

        This is a convenience so we don't need to import the parameter class,
        to instantiate an intervenor class it is associated with.

        !!! Example
            ```python
            CurlField.with_params(amplitude=10.0)
            ```
        """
        param_cls = next((f for f in fields(cls) if f.name == "params")).type
        param_fields = [f.name for f in fields(param_cls)]

        return cls(
            param_cls(**{k: v for k, v in kwargs.items() if k in param_fields}),  # type: ignore
            **{k: v for k, v in kwargs.items() if k not in param_fields},
        )

    # @cached_property
    # def _broadcast_output(self, substate, other):
    #     if not jtu.tree_structure(substate) == jtu.tree_structure(other):
    #         if (
    #             isinstance(other, ArrayLike)
    #             and isinstance(substate, PyTree[type(other)])
    #         ):
    #             return lambda substate, other: jt.map(lambda _: other, substate)
    #         else:
    #             raise ValueError("Intervenor output structure 1) does not match "
    #                              "indicated output substate, and 2) is not a "
    #                              "single `ArrayLike` that can be broadcast to "
    #                              "match the output substate")
    #     else:
    #         return lambda substate, other: other

    def __call__(self, input: InputT, state: StateT, *, key: PRNGKeyArray) -> StateT:
        """Return a state PyTree modified by the intervention.

        Arguments:
            input: PyTree of intervention parameters. If any leaves are `None`, they
                will be replaced by the corresponding leaves of `self.params`.
            state: The model state to be intervened upon.
            key: A key to provide randomness for the intervention.
        """
        params: InputT = eqx.combine(input, self.params)

        def _get_updated_substate():
            substate = self.out_where(state)
            other = self.transform(params, self.in_where(state), key=key)
            return jt.map(
                lambda x, y: self.operation(x, params.scale * y),
                substate,
                other,
            )

        return jax.lax.cond(
            params.active,
            lambda: eqx.tree_at(  # Replace the `out_where` substate
                self.out_where,
                state,
                _get_updated_substate(),
            ),
            lambda: state,
        )

    @abstractmethod
    def transform(
        self,
        params: InputT,
        substate_in: PyTree[ArrayLike, " T"],
        *,
        key: PRNGKeyArray,
    ) -> PyTree[ArrayLike, " S"]:
        """Transforms the input substate to produce an altered output substate."""
        ...


# InputsT = TypeVar("InputsT", bound=Sequence[AbstractIntervenorInput])


# class MultiIntervenor(Generic[StateT, InputsT]):
#     component_params: InputsT
#     operation: Callable[[ArrayLike, ArrayLike], ArrayLike] = op.add

#     def __call__(self, inputs: InputsT, state: StateT, *, key: PRNGKeyArray) -> StateT:


class CurlFieldParams(AbstractIntervenorInput):
    """Parameters for a curl force field.

    Attributes:
        scale: Scaling factor on the intervenor output.
        active: Whether the force field is active.
        amplitude: The amplitude of the force field. Negative is clockwise, positive
            is counterclockwise.
    """

    scale: float = 1.0
    amplitude: float = 1.0
    active: bool = True


class CurlField(AbstractIntervenor["MechanicsState", CurlFieldParams]):
    """Apply a curl force field to a mechanical effector.

    Attributes:
        params: Default curl field parameters.
        in_where: Returns the substate corresponding to the effector's velocity.
        out_where: Returns the substate corresponding to the force on the effector.
        operation: How to combine the effector force due to the curl field,
            with the existing force on the effector. Default is addition.
    """

    params: CurlFieldParams = CurlFieldParams()
    in_where: Callable[["MechanicsState"], Float[Array, "... ndim=2"]] = (
        lambda state: state.effector.vel
    )
    out_where: Callable[["MechanicsState"], Float[Array, "... ndim=2"]] = (
        lambda state: state.effector.force
    )
    operation: Callable[[ArrayLike, ArrayLike], ArrayLike] = op.add

    def transform(
        self,
        params: CurlFieldParams,
        substate_in: Float[Array, "ndim=2"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "ndim=2"]:
        """Transform velocity into curl force."""
        return params.amplitude * jnp.array([-1, 1]) * substate_in[..., ::-1]


class FixedFieldParams(AbstractIntervenorInput):
    """Parameters for a fixed force field.

    Attributes:
        amplitude: The scale of the force field.
        active: Whether the force field is active.
    """

    scale: float = 1.0
    amplitude: float = 1.0
    field: Float[Array, "ndim=2"] = field(default_factory=lambda: jnp.array([0.0, 0.0]))
    active: bool = True


class FixedField(AbstractIntervenor["MechanicsState", FixedFieldParams]):
    """Apply a fixed (state-independent) force field to a mechanical effector.

    Attributes:
        params: Default force field parameters.
        in_where: Unused.
        out_where: Returns the substate corresponding to the force on the effector.
        operation: How to combine the effector force due to the fixed field,
            with the existing force on the effector. Default is addition.
    """

    params: FixedFieldParams = FixedFieldParams()
    in_where: Callable[["MechanicsState"], Any] = lambda state: state.effector
    out_where: Callable[["MechanicsState"], Float[Array, "... ndim=2"]] = (
        lambda state: state.effector.force
    )
    operation: Callable[[ArrayLike, ArrayLike], ArrayLike] = op.add

    def transform(
        self,
        params: FixedFieldParams,
        substate_in: Float[Array, "ndim=2"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "ndim=2"]:
        """Return the scaled force."""
        return params.amplitude * params.field


class AddNoiseParams(AbstractIntervenorInput):
    """Parameters for adding noise to a state.

    Attributes:
        scale: Constant factor to multiply the noise samples.
        noise_func: A function that returns noise samples.
        active: Whether the intervention is active.
    """

    scale: float = 1.0
    active: bool = True


class AddNoise(AbstractIntervenor[StateT, AddNoiseParams]):
    """Add noise to a part of the state.

    Attributes:
        params: Default intervention parameters.
        out_where: Returns the substate to which noise is added.
        operation: How to combine the noise with the substate. Default is
            addition.
    """

    params: AddNoiseParams = AddNoiseParams()
    noise_func: Callable[[PRNGKeyArray, Array], Array] = Normal()
    in_where: Callable[[StateT], PyTree[Array, " T"]] = lambda state: state
    out_where: Callable[[StateT], PyTree[Array, " T"]] = lambda state: state
    operation: Callable[[ArrayLike, ArrayLike], ArrayLike] = op.add

    def transform(
        self,
        params: AddNoiseParams,
        substate_in: PyTree[Array, " T"],
        *,
        key: PRNGKeyArray,
    ) -> PyTree[Array, " T"]:
        """Return a PyTree of scaled noise arrays with the same structure/shapes as
        `substate_in`."""
        noise = jt.map(
            lambda x: self.noise_func(key, x),
            substate_in,
        )
        return noise


class NetworkIntervenorParams(AbstractIntervenorInput):
    """Parameters for interventions on network unit activity.

    Attributes:
        unit_spec: A PyTree of arrays with the same tree structure and array shapes
            as the substate of the network to be intervened upon, specifying the
            unit-wise activities that constitute the perturbation.
        active: Whether the intervention is active.

    !!! Note ""
        Note that `unit_spec` may be a single array—which is just a single-leaf PyTree
        of arrays—when `out_where` of the intervenor is also a single array.
    """

    scale: float = 1.0
    unit_spec: Optional[PyTree] = None
    active: bool = True


class NetworkClamp(AbstractIntervenor["NetworkState", NetworkIntervenorParams]):
    """Clamps some of a network's units' activities to given values.

    Attributes:
        params: Default intervention parameters.
        out_where: Returns the substate of arrays giving the activity of the units
            whose activities may be clamped.
        operation: How to combine the original and clamped unit activities. Default
            is to replace the original with the altered.
    """

    params: NetworkIntervenorParams = NetworkIntervenorParams()
    in_where: Callable[["NetworkState"], PyTree[Array, " T"]] = lambda state: state.hidden
    out_where: Callable[["NetworkState"], PyTree[Array, " T"]] = lambda state: state.hidden
    operation: Callable[[ArrayLike, ArrayLike], ArrayLike] = lambda x, y: y

    def transform(
        self,
        params: NetworkIntervenorParams,
        substate_in: PyTree[Array, " T"],
        *,
        key: PRNGKeyArray,
    ) -> PyTree[Array, " T"]:
        return jt.map(
            lambda x, y: jnp.where(jnp.isnan(y), x, y),
            substate_in,
            params.unit_spec,
        )


class NetworkConstantInput(AbstractIntervenor["NetworkState", NetworkIntervenorParams]):
    """Adds a constant input to some network units.

    Attributes:
        params: Default intervention parameters.
        out_where: Returns the substate of arrays giving the activity of the units
            to which a constant input may be added.
        operation: How to combine the original and altered unit activities. Default
            is addition.
    """

    params: NetworkIntervenorParams = NetworkIntervenorParams()
    in_where: Callable[["NetworkState"], PyTree[Array, " T"]] = lambda state: state.hidden
    out_where: Callable[["NetworkState"], PyTree[Array, " T"]] = lambda state: state.hidden
    operation: Callable[[ArrayLike, ArrayLike], ArrayLike] = op.add

    def transform(
        self,
        params: NetworkIntervenorParams,
        substate_in: "NetworkState",
        *,
        key: PRNGKeyArray,
    ) -> PyTree[Array, " T"]:
        return jt.map(jnp.nan_to_num, params.unit_spec)


class ConstantInputParams(AbstractIntervenorInput):
    """Parameters for adding a constant input to a state array.

    Attributes:
        scale: Constant factor to multiply the input.
        arrays: A PyTree of arrays with the same tree structure and array shapes
            as the substate of the state to be intervened upon, specifying the
            constant input to be added.
        active: Whether the intervention is active.
    """

    scale: float = 1.0
    arrays: Optional[PyTree] = ()
    active: bool = True


class ConstantInput(AbstractIntervenor[StateT, ConstantInputParams]):
    """Adds a constant input to a state array.

    Attributes:
        params: Default intervention parameters.
        out_where: Returns the substate of arrays to which a constant input is added.
        operation: How to combine the original and altered substates. Default is addition.
    """

    params: ConstantInputParams = ConstantInputParams()
    in_where: Callable[[StateT], PyTree[Array, " T"]] = lambda state: state
    out_where: Callable[[StateT], PyTree[Array, " T"]] = lambda state: state
    operation: Callable[[ArrayLike, ArrayLike], ArrayLike] = op.add

    def transform(
        self,
        params: ConstantInputParams,
        substate_in: PyTree[Array, " T"],
        *,
        key: PRNGKeyArray,
    ) -> PyTree[Array, " T"]:
        return params.arrays


class CopyParams(AbstractIntervenorInput):
    scale: float = 1.0
    active: bool = True


class Copy(AbstractIntervenor[StateT, CopyParams]):
    in_where: Callable[[StateT], PyTree[Array, " T"]]
    out_where: Callable[[StateT], PyTree[Array, " T"]]
    operation: Callable[[ArrayLike, ArrayLike], ArrayLike] = lambda x, y: y
    params: CopyParams = CopyParams()

    def transform(
        self,
        params: CopyParams,
        substate_in: PyTree[Array, " T"],
        *,
        key: PRNGKeyArray,
    ) -> PyTree[Array, " T"]:
        return substate_in


def is_intervenor(element: Any) -> bool:
    """Return `True` if `element` is an Intervenor."""
    return isinstance(element, AbstractIntervenor)
