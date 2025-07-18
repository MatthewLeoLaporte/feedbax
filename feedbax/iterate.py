"""Models that iterate other models.

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
import os
from typing import Optional, Tuple

import equinox as eqx
from equinox import AbstractVar, Module
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array, PRNGKeyArray, PyTree, Shaped

from feedbax._progress import _tqdm
from feedbax._model import AbstractModel
from feedbax.state import StateT
from feedbax._tree import tree_take, tree_set

logger = logging.getLogger(__name__)


class AbstractIterator(AbstractModel[StateT]):
    """Base class for models which iterate other models."""

    _step: AbstractVar[AbstractModel[StateT]]
    n_steps: AbstractVar[int]

    def init(self, *, key: PRNGKeyArray) -> StateT:
        """Returns an initial state for the iterated model."""
        return self._step.init(key=key)

    @property
    def step(self) -> Module:
        """The model to be iterated."""
        return self._step

    def state_consistency_update(self, state: StateT) -> StateT:
        return self._step.state_consistency_update(state)


class Iterator(AbstractIterator[StateT]):
    """Applies a model repeatedly, carrying state. Returns history for all states.

    !!! NOTE
        If memory is not an issue, this class is preferred to
        `ForgetfulIterator` as it lacks the state partitioning overhead, and
        is therefore faster.

        For very large state PyTrees, however, it may be preferable to use
        `ForgetfulIterator` to save memory.

    Attributes:
        step (AbstractModel[StateT]): The model to be iterated.
        n_steps: The number of steps to iterate for.
    """

    _step: AbstractModel[StateT]
    n_steps: int

    def __init__(
        self,
        step: AbstractModel[StateT],
        n_steps: int,
    ):
        """
        Arguments:
            step: The model to be iterated.
            n_steps: The number of steps to iterate for.
        """
        self._step = step
        self.n_steps = n_steps

    def __call__(
        self,
        input_: PyTree,
        state: StateT,
        key: PRNGKeyArray,
    ) -> StateT:
        """
        Arguments:
            input_: The input to the model.
            state: The initial state of the model to be iterated.
            key: Determines the pseudo-randomness in model execution.
        """

        keys = jr.split(key, self.n_steps - 1)

        def step(state, args):
            input, key = args
            state = self._step(input, state, key)
            return state, state

        _, states = lax.scan(
            step,
            state,
            (input_, keys),
        )

        return jt.map(
            lambda state0, state: jnp.concatenate([state0[None], state], axis=0),
            state,
            states,
        )


class ForgetfulIterator(AbstractIterator[StateT]):
    """Applies a model repeatedly, carrying state. Returns history for a subset of states.

    Attributes:
        n_steps: The number of steps to iterate for.
        step: The model to be iterated.
        memory_spec: A PyTree of bools—a prefix of `StateT` indicating which states to store.
    """

    _step: AbstractModel[StateT]
    n_steps: int
    memory_spec: PyTree[bool]

    def __init__(
        self,
        step: AbstractModel[StateT],
        n_steps: int,
        memory_spec: Optional[PyTree[bool]] = None,
    ):
        """
        Arguments:
            step: The model to be iterated.
            n_steps: The number of steps to iterate for.
            memory_spec: A PyTree of bools indicating which states to store.
        """
        if memory_spec is None:
            memory_spec = step.memory_spec
        self._step = step
        self.n_steps = n_steps
        self.memory_spec = memory_spec

    @jax.named_scope("fbx.ForgetfulIterator")
    def __call__(
        self,
        input_: PyTree,  # [Shaped[Array, "n_steps ..."]],
        state: StateT,
        key: PRNGKeyArray,
    ) -> StateT:  #! Adds a batch dimension, actually.
        """
        Arguments:
            input_: The input to the model.
            state: The initial state of the model to be iterated.
            key: Determines the pseudo-randomness in model execution.
        """
        key1, key2, key3 = jr.split(key, 3)

        init_input = tree_take(input_, 0)
        states = self._init_arrays(init_input, state, key2)

        if os.environ.get("FEEDBAX_DEBUG", False) == "True":
            for i in _tqdm(range(self.n_steps), desc="steps"):
                input_, states, key3 = self._body_func(i, (input_, states, key3))

            return states

        _, states, _ = lax.fori_loop(
            0,
            self.n_steps,
            self._body_func,
            (input_, states, key3),
        )

        return states

    @jax.named_scope("fbx.ForgetfulIterator._body_func")
    def _body_func(self, i: int, x: Tuple) -> Tuple:
        inputs, states, key = x

        key1, key2 = jr.split(key)

        # Since we optionally store the trajectories of only some of the states,
        # as specified by `memory_spec`, we need to partition these out
        # so we can index them, then recombine with the states for which only
        # a single time step (the current one) is stored.
        states_mem, state_nomem = eqx.partition(states, self.memory_spec)
        state_mem, input = tree_take((states_mem, inputs), i)
        state = eqx.combine(state_mem, state_nomem)

        state = self._step(input, state, key1)

        # Likewise, split the resulting states into those which are stored,
        # which are then assigned to the next index in the trajectory, and
        # recombined with the single-timestep states.
        state_mem, state_nomem = eqx.partition(state, self.memory_spec)
        states_mem = tree_set(states_mem, state_mem, i + 1)
        states = eqx.combine(states_mem, state_nomem)

        return inputs, states, key2

    @jax.named_scope("fbx.ForgetfulIterator.init_arrays")
    def _init_arrays(
        self,
        input: PyTree,
        init_state: StateT,
        key: PRNGKeyArray,
    ) -> StateT:  #! Adds a batch dimension re: `init_state`
        """Returns a model state PyTree with a batch dimension added to store
        history on each iteration, but only for those array leaves specified
        by `self.memory_spec`.

        The shape of the arrays in the PyTree(s) returned by `step` is
        automatically inferred, and used to initialize empty history arrays
        in which to store the states across all steps; `memory_spec` can be
        used to specify which states to store. By default, all are stored.
        """
        # Get the shape of the state output by `self._step`
        outputs = eqx.filter_eval_shape(
            self._step,
            input,
            init_state,
            key=key,
        )

        # Generate empty trajectories for mem states
        scalars, array_structs = eqx.partition(
            eqx.filter(outputs, self.memory_spec),
            eqx.is_array_like,  # False for jax.ShapeDtypeStruct
        )
        asarrays = eqx.combine(jt.map(jnp.asarray, scalars), array_structs)
        states = jt.map(
            lambda x: jnp.zeros((self.n_steps, *x.shape), dtype=x.dtype),
            asarrays,
        )

        # Insert the init state for mem states; combine with no mem state
        init_state_mem, init_state_nomem = eqx.partition(init_state, self.memory_spec)
        states = eqx.combine(tree_set(states, init_state_mem, 0), init_state_nomem)

        return states


def eval_state_traj(
    #! should be any callable (input, state, *, key) -> state
    model: AbstractModel[StateT],
    state0: StateT,
    n_steps: int,
    input: PyTree[Array],
    key: PRNGKeyArray,
):
    """Evaluate the state trajectory of a model with fixed inputs.
    """
    keys = jr.split(key, n_steps - 1)
    # Assume constant inputs.
    inputs = jnp.broadcast_to(input, (n_steps - 1, *input.shape))

    def step(state, args):
        input, key = args
        #! key=key won't work with AbstractModel in general, yet
        state = model(input, state, key=key)
        return state, state

    _, states = jax.lax.scan(
        step,
        state0,
        (inputs, keys),
    )

    return jnp.concatenate([
        state0[None], states
    ])