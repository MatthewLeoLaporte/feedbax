"""Base classes for stateful models.

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
from collections.abc import Callable, Mapping
from functools import wraps
import logging
from typing import (
    TYPE_CHECKING,
    Generic,
    Optional,
    Self,
)

import equinox as eqx
from equinox import Module
import jax
from jaxtyping import Array, PRNGKeyArray, PyTree
import numpy as np

from feedbax.misc import is_module
from feedbax.state import StateBounds, StateT
from feedbax._tree import random_split_like_tree

if TYPE_CHECKING:
    from feedbax._staged import AbstractStagedModel
    from feedbax.intervene import AbstractIntervenorInput


logger = logging.getLogger(__name__)


class AbstractModel(Module, Generic[StateT]):
    """Base class for all Feedbax models."""

    @abstractmethod
    def __call__(
        self,
        input_: PyTree[Array],
        state: StateT,
        key: PRNGKeyArray,
    ) -> StateT:
        """Return an updated state, given inputs and a prior state.

        Arguments:
            input_: The inputs to the model.
            state: The prior state associated with the model.
            key: A random key used for model operations.
        """
        ...

    @abstractproperty
    def step(self) -> Module:
        """The part of the model PyTree specifying a single time step of the model.

        For non-iterated models, this should trivially return `step`.
        """
        ...

    def state_consistency_update(
        self,
        state: StateT,
    ) -> StateT:
        """Make sure the model state is self-consistent.

        !!! Info
            The default behaviour is to just return the same state that was passed.

            In some models, multiple representations of the same information are kept,
            but only one is initialized by the task. A typical example is a task that
            initializes a model's effector state, e.g. the location of the endpoint of
            the arm at the start of a reach. In the case of a two-link arm, the location
            of the arm's endpoint directly constrains the joint angles. For the state to
            be consistent, the values for the joint angles should match the values for
            effector position. Importantly, the joint angles are the representation that
            is used by the ODE describing the arm. Therefore we call
            `state_consistency_update` after initializing the state and before the first
            forward pass the model, to make sure the joint angles are consistent with
            the effector position.

            In this way we avoid having to specify redundant information in
            `AbstractTask`, and each model can handle the logic of what makes
            a state consistent, with respect to its own operations.
        """
        return state

    @abstractmethod
    def init(
        self,
        *,
        key: PRNGKeyArray,
    ) -> StateT:
        """Return a default state for the model."""
        ...

    @property
    def bounds(self) -> PyTree[StateBounds]:
        """Suggested bounds on the state variables."""
        return StateBounds(low=None, high=None)

    @property
    def memory_spec(self) -> PyTree[bool]:
        """Specifies which states should typically be remembered.

        !!! Info
            This is not used by the model itself, but may be used by an
            `AbstractIterator` model that wraps it. When iterating very large models for
            many time steps, storing all states may be limiting because of available
            memory; not storing certain parts of the state across all time steps may
            be helpful.
        """
        return True


class ModelInput(Module):
    """PyTree that contains all inputs to a model."""

    value: PyTree[Array]
    intervene: Mapping[str, "AbstractIntervenorInput"]


class MultiModel(AbstractModel[StateT]):

    models: PyTree[AbstractModel[StateT], "T"]

    def __call__(
        self,
        input_: ModelInput,
        state: PyTree[StateT, "T"],
        key: PRNGKeyArray,
    ) -> StateT:

        # TODO: This is hacky, because I want to pass intervenor stuff through entirely. See `staged`
        return jax.tree_map(
            lambda model, input_value, state, key: model(
                ModelInput(input_value, input_.intervene), state, key
            ),
            self.models,
            input_.value,
            state,
            self._get_keys(key),
            is_leaf=lambda x: isinstance(x, AbstractModel),
        )

    def __getitem__(self, idx):
        return jax.tree_util.tree_leaves(self.models, is_leaf=is_module)[idx]

    def init(
        self,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> StateT:
        return jax.tree_map(
            lambda model, key: model.init(key=key),
            self.models,
            self._get_keys(key),
            is_leaf=lambda x: isinstance(x, AbstractModel),
        )

    @property
    def step(self) -> Module:
        return self

    def _get_keys(self, key):
        return random_split_like_tree(
            key, tree=self.models, is_leaf=lambda x: isinstance(x, AbstractModel)
        )


def wrap_stateless_callable(callable: Callable):
    """Makes a 'stateless' callable compatible with state-passing.

    !!! Info
        `AbstractStagedModel` defines its operations as transformations to parts of
        a state PyTree. Each stage of a model consists of passing a particular substate
        of the model to a callable that operates on it, returning an updated substate.
        However, in some cases the new substate does not depend on the previous
        substate, but is generated entirely from some other inputs.

        For example, a linear neural network layer outputs an array of a certain shape,
        but only requires some input array—and not its prior output (state) array—to
        do so. We can use a module like `eqx.nn.Linear` to update a part of a model's
        state, as the callable of one of its model stages; however, the signature of
        `Linear` only accepts `input`, and not `state`. By wrapping in this function, we
        can make it accept `state` as well, though it is simply discarded.

    Arguments:
        callable: The callable to wrap.
    """
    @wraps(callable)
    def wrapped(input, state, *args, **kwargs):
        return callable(input, *args, **kwargs)

    return wrapped

def wrap_stateless_keyless_callable(callable: Callable):
    """Like `wrap_stateless_callable`, for a callable that also takes no `key`.

    Arguments:
        callable: The callable to wrap.
    """
    @wraps(callable)
    def wrapped(input, state, *args, key: Optional[PRNGKeyArray] = None, **kwargs):
        return callable(input, *args, **kwargs)
    return wrapped


