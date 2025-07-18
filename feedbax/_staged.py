"""Base classes for stateful models with stages.

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
import logging
import os
from typing import (
    TYPE_CHECKING,
    Generic,
    Optional,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    Union, dataclass_transform,
)

import equinox as eqx
from equinox import AbstractVar, Module, field
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree
import numpy as np

from feedbax._model import AbstractModel, ModelInput
from feedbax.intervene import AbstractIntervenor
from feedbax.intervene.intervene import AbstractIntervenorInput
from feedbax.intervene.schedule import (
    ArgIntervenors, 
    Intervenor, 
    IntervenorLabelStr, 
    ModelIntervenors, 
    StageIntervenors, 
    _fixed_intervenor_label, 
    pre_first_stage,
    post_final_stage,
)
from feedbax.misc import indent_str, is_module
from feedbax.state import StateT


logger = logging.getLogger(__name__)


ModelT = TypeVar("ModelT", bound=Module)
T = TypeVar("T", bound=Module)

class ModelStageCallable(Protocol):
    # This is part of the `ModelInput` hack.
    def __call__(self, input_: ModelInput, state: PyTree[Array], *, key: PRNGKeyArray) -> PyTree[Array]:
        ...
        

class OtherStageCallable(Protocol):
    def __call__(self, input_: PyTree[Array], state: PyTree[Array], *, key: PRNGKeyArray) -> PyTree[Array]:
        ...


class ModelStage(Module, Generic[ModelT, T]):
    """Specification for a stage in a subclass of `AbstractStagedModel`.

    Each stage of a model is a callable that performs a modification to part
    of the model state.

    !!! Note
        To ensure that references to parts of the model instance remain fresh,
        `callable_` takes the instance of `AbstractStagedModel` (i.e. `self`)
        and returns the callable associated with the stage.

        It is possible for references to become stale. For example, if we
        assign `callable_=self.net` for the neural network update in
        [`SimpleFeedback`][feedbax.bodies.SimpleFeedback], then it will
        continue to refer to the neural network assigned to `self.net`
        upon the model's construction, even after the network weights
        have been updated during training—so, the model will not train.

    Attributes:
        callable_: The module, method, or function that transforms part of the
            model state.
        where_input: Selects the  parts of the input and state to be passed
            as input to `callable_`.
        where_state: Selects the substate that passed and return as state to
            `callable_`.
        intervenors: Optionally, a sequence of state interventions to be
            applied at the beginning of this model stage.
    """

    callable: Callable[
        [ModelT],
        Union[ModelStageCallable, OtherStageCallable],
    ]
    where_input: Callable[[PyTree, T], PyTree]
    where_state: Callable[[T], PyTree]
    intervenors: StageIntervenors[T] = field(default_factory=tuple)


StageNameStr: TypeAlias = str


class AbstractStagedModel(AbstractModel[StateT]):
    """Base class for state-dependent models whose stages can be intervened upon.

    !!! Info
        To define a new staged model, the following complementary components
        must be implemented:

        1. A PyTree of model states -- typically, a
           [final](https://docs.kidger.site/equinox/pattern/) subclass of
           `equinox.Module`. The fields of the PyTree are typically JAX
           arrays, or else other PyTrees of model states associated with the
           model's components.
        2. A final subclass of
           [`AbstractStagedModel`][feedbax.AbstractStagedModel]. Note that the
           abstract class is a `Generic`, and for proper type checking, the
           type argument of the subclass should be the type of state PyTree
           defined in (1).

            This subclass must implement the following:

            1. A `model_spec` property giving a mapping from stage labels
               to [`ModelStage`][feedbax.ModelStage] instances, each
               specifying an operation performed on the model state.
            2. An `init` method that takes a random key and returns a default
               model state.

        For an example, consider 1) [`SimpleFeedbackState`][feedbax.bodies.SimpleFeedbackState]
        and 2) [`SimpleFeedback`][feedbax.bodies.SimpleFeedback].
    """

    intervenors: AbstractVar[ModelIntervenors[StateT]]

    def __call__(
        self,
        input_: ModelInput,
        state: StateT,
        key: PRNGKeyArray,
    ) -> StateT:
        """Return an updated model state, given input and a prior state.

        Arguments:
            input_: The input to the model.
            state: The prior state of the model.
            key: A random key which will be split to provide separate keys for
                each model stage and intervenor.
        """
        with jax.named_scope(type(self).__name__):

            # Intervenors may be scheduled prior to the first model stage.
            if pre_first_stage in self.intervenors:
                state = self._apply_intervenors(
                    self.intervenors[pre_first_stage],
                    input_.intervene,
                    state,
                    key,
                )

            keys = jr.split(key, len(self._stages))

            for (label, stage), key in zip(self._stages.items(), keys):

                key_intervene, key_stage = jr.split(key)

                callable_ = stage.callable(self)
                subinput = stage.where_input(input_.value, state)

                # TODO: What's a less hacky way of doing this?
                # I was trying to avoid introducing additional parameters to `AbstractStagedModel.__call__`
                if isinstance(callable_, AbstractModel):
                    callable_input = ModelInput(subinput, input_.intervene)
                else:
                    callable_input = subinput

                state = eqx.tree_at(
                    stage.where_state,
                    state,
                    callable_(
                        callable_input,
                        stage.where_state(state),
                        key=key_stage,
                    ),
                )

                # Intervenors assigned to a stage are applied after the stage.
                state = self._apply_intervenors(
                    stage.intervenors,
                    input_.intervene,
                    state,
                    key_intervene,
                )

                if os.environ.get("FEEDBAX_DEBUG", False) == "True":
                    debug_strs = [
                        indent_str(eqx.tree_pformat(x), indent=4)
                        for x in (callable_, subinput, stage.where_state(state))
                    ]

                    log_str = "\n".join(
                        [
                            f"Model type: {type(self).__name__}",
                            f'Stage: "{label}"',
                            f"Callable:\n{debug_strs[0]}",
                            f"Input:\n{debug_strs[1]}",
                            f"Substate:\n{debug_strs[2]}",
                        ]
                    )

                    logger.debug(f"\n{indent_str(log_str, indent=2)}\n")
                    
            # Intervenors may also be explicitly scheduled for after the final model stage.
            if post_final_stage in self.intervenors:
                state = self._apply_intervenors(
                    self.intervenors[post_final_stage],
                    input_.intervene,
                    state,
                    key,
                )

        return state

    # This is a method rather than a function so we can get the generic `StateT` typing
    def _apply_intervenors(
        self,
        stage_intervenors: Mapping[IntervenorLabelStr, AbstractIntervenor],
        params: Mapping[str, AbstractIntervenorInput],
        state: StateT,
        key: PRNGKeyArray,
    ) -> StateT:
        keys_intervene = jr.split(key, len(stage_intervenors))

        for (label, intervenor), k in zip(stage_intervenors.items(), keys_intervene):
            if label.startswith("FIXED") or label not in params:
                # Fixed (trial-invariant) intervenor -- or, no params supplied
                state = intervenor(None, state, key=k)
            else:
                # Per-trial params provided by the task.
                state = intervenor(params[label], state, key=k)
        return state

    @abstractmethod
    def init(
        self,
        *,
        key: PRNGKeyArray,
    ) -> StateT:
        """Return a default state for the model."""
        ...

    @abstractproperty
    def model_spec(self) -> OrderedDict[str, ModelStage[Self, StateT]]:
        """Specify the model's computation in terms of state operations.

        !!! Warning
            It's necessary to return `OrderedDict` because `jax.tree_util`
            still sorts `dict` keys, which usually puts the stages out of order.
        """
        ...

    @cached_property
    def _stages(self) -> OrderedDict[str, ModelStage[Self, StateT]]:
        """Zips up the user-defined intervenors with `model_spec`.

        This should not be referred to in `__init__` before assigning `self.intervenors`!
        """

        return jax.tree_map(
            lambda x, y: eqx.tree_at(lambda x: x.intervenors, x, y),
            self.model_spec,
            OrderedDict({
                k: self.intervenors.get(k, {}) for k in self.model_spec
            }),
            is_leaf=lambda x: isinstance(x, ModelStage),
        )

    def _get_intervenors_dict(
        self,
        intervenors: Optional[ArgIntervenors[StateT]],
    ) -> ModelIntervenors[StateT]:
        """Specifically for fixed intervenors."""

        intervenors_dict = {}

        if intervenors is not None:
            if isinstance(intervenors, Sequence):
                # By default, place interventions before the first stage.
                intervenors_dict |= {pre_first_stage:
                    OrderedDict({
                        _fixed_intervenor_label(intervenor): intervenor
                        for intervenor in intervenors
                    })
                }
            elif isinstance(intervenors, Mapping):
                intervenors_dict |= {
                    stage_name: OrderedDict({
                        _fixed_intervenor_label(intervenor): intervenor
                        for intervenor in stage_intervenors
                    })
                    for stage_name, stage_intervenors in intervenors.items()
                }
            else:
                raise ValueError("intervenors not a sequence or dict of sequences")

        # It's necessary to use `OrderedDict` to keep the `Optional[str]` type for
        # keys; otherwise JAX will try to do sorting operations on `None` vs. strings
        return intervenors_dict

    @property
    def step(self) -> Module:
        """The model step.

        For an `AbstractStagedModel`, this is trivially the model itself.
        """
        return self

    # TODO: Avoid referencing `AbstractIntervenor` here, to avoid a circular import
    # with `feedbax.intervene`.
    @property
    def _all_intervenor_labels(self):
        model_leaves_with_paths = jax.tree_util.tree_leaves_with_path(
            self, is_leaf=lambda x: isinstance(x, AbstractIntervenor)
        )
        labels = [
            path[-1].key for path, leaf in model_leaves_with_paths
            if isinstance(leaf, AbstractIntervenor)
        ]
        return tuple(labels)


def pformat_model_spec(
    model: AbstractStagedModel,
    indent: int = 2,
    newlines: bool = False,
) -> str:
    """Returns a string representation of the model specification tree.

    Shows what is called by `model`, and by any `AbstractStagedModel`s it calls.

    !!! Warning
        This assumes that the model spec is a tree/DAG. If there are cycles in
        the model spec, this will recurse until an exception is raised.

    Arguments:
        model: The staged model to format.
        indent: Number of spaces to indent each nested level of the tree.
        newlines: Whether to add an extra blank line between each line.
    """

    def get_spec_strs(model: AbstractStagedModel):
        spec_strs = []

        for label, stage_spec in model._stages.items():
            intervenor_str = "".join(
                [
                    f"intervenor: {type(intervenor).__name__}\n"
                    for intervenor in stage_spec.intervenors
                ]
            )

            callable = stage_spec.callable(model)

            spec_str = f"{label}: "

            if getattr(callable, "__wrapped__", None) is not None:
                spec_str += "wrapped: "
                # callable = callable.__wrapped__

            # BoundMethods
            if (func := getattr(callable, "__func__", None)) is not None:
                owner = type(getattr(callable, "__self__")).__name__
                spec_str += f"{owner}.{func.__name__}"
            # Functions
            elif (name := getattr(callable, "__name__", None)) is not None:
                spec_str += f"{name}"
            # Modules and other callable instances
            else:
                spec_str += f"{type(callable).__name__}"

            spec_strs += [intervenor_str + spec_str]

            if isinstance(callable, AbstractStagedModel):
                spec_strs += [
                    " " * indent + spec_str for spec_str in get_spec_strs(callable)
                ]

        return spec_strs

    nl = "\n\n" if newlines else "\n"

    return nl.join(get_spec_strs(model))


def pprint_model_spec(
    model: AbstractStagedModel,
    indent: int = 2,
    newlines: bool = False,
):
    """Prints a string representation of the model specification tree.

    Shows what is called by `model`, and by any `AbstractStagedModel`s it calls.

    !!! Warning
        This assumes that the model spec is a tree. If there are cycles in
        the model spec, this will recurse until an exception is raised.

    Arguments:
        model: The staged model to format.
        indent: Number of spaces to indent each nested level of the tree.
        newlines: Whether to add an extra blank line between each line.
    """
    print(pformat_model_spec(model, indent=indent, newlines=newlines))
