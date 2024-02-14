"""Models that iterate other state-updating models.

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
import os
from typing import Generic, Optional, Tuple, TypeVar

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree, Shaped
from tqdm.auto import tqdm

from feedbax.model import AbstractModel
from feedbax.state import AbstractState
from feedbax.tree import tree_take, tree_set

logger = logging.getLogger(__name__)


StateT = TypeVar("StateT", bound=AbstractState)
# ModelT = TypeVar("ModelT", bound=AbstractModel)


class AbstractIterator(AbstractModel[StateT]):
    """Base class for models which iterate other models.
    
    A main responsibility of this class, aside from defining abstract fields, 
    is to expose the interface of the iterated model, since we are often 
    interested in the methods belonging to the model defined as a single step. 
    
    Some issues with inheriting from `AbstractModel`: 
    - `ForgetfulIterator.__call__` adds a batch dimension to the returned `StateT`.
      Technically I don't think we can (should?) type this differently anyway,
      but in principle it is not the same type.
    - `AbstractModel` is a generic over a type variable `StateT` of
      `AbstractState`; whereas `AbstractIterator` should be a generic over
      `AbstractModel[StateT]`, I think. I don't know that there is a way to achieve
      this without higher-kinded types. 
      - We could make this class a generic of both `AbstractModel[StateT]` 
        and a `ModelT`, but I don't think there's a way to associate them and 
        enforce that `ModelT` is a generic of `StateT`.
      - I think `AbstractModel` should keep being a generic of `StateT`, because
        of the signatures of its methods.
    """
    
    step: AbstractVar[AbstractModel[StateT]]  
    n_steps: AbstractVar[int]

    def init(self, *, key: PRNGKeyArray) -> StateT:
        """Initialize the state of the iterated model.
        """
        return self.step.init(key=key)
    
    @property 
    def _step(self):
        return self.step


class Iterator(AbstractIterator[StateT]):
    """Applies a model repeatedly, carrying state. Returns history for all states.
    
    !!! NOTE
        If memory is not an issue, this class is preferred to
        `ForgetfulIterator` as it lacks the overhead of state partitioning, and
        is therefore faster. 
        
        For very large state PyTrees, however, it may be preferable to use
        `ForgetfulIterator` to save memory.
    """
    step: AbstractModel[StateT]
    n_steps: int 
    
    def __init__(self, step: AbstractModel[StateT], n_steps: int):
        """**Arguments**:
        
        - `step` is the model to be iterated.
        - `n_steps` is the number of steps to iterate for.
        """
        self.step = step
        self.n_steps = n_steps
    
    def __call__(
        self, 
        input,
        state: StateT, 
        key: PRNGKeyArray,
    ) -> StateT:
        """
        **Arguments**:
        
        - `input` is the 
        """
        
        keys = jr.split(key, self.n_steps - 1)
        
        def step(state, args): 
            input, key = args
            state = self.step(input, state, key)
            return state, state
                
        _, states = lax.scan(
            step,
            state, 
            (input, keys),
        )

        return jax.tree_map(
            lambda state0, state: jnp.concatenate([state0[None], state], axis=0),
            state,
            states,
        )


class ForgetfulIterator(AbstractIterator[StateT]):
    """Applies a model repeatedly, carrying state. Returns history for a subset of states.
    """
    step: AbstractModel[StateT]
    n_steps: int 
    states_includes: PyTree[bool]  # can't do StateT[bool]
    
    def __init__(
        self,
        step: AbstractModel[StateT],
        n_steps: int,
        states_includes: Optional[PyTree[bool]] = None,
    ):
        """**Arguments**:
        
        - `step`: is the model to be iterated.
        - `n_steps`: is the number of steps to iterate for.
        - `states_includes`: is a PyTree of bools indicating which states to store.
        """
        if states_includes is None:
            states_includes = step.memory_spec
        self.step = step
        self.n_steps = n_steps
        self.states_includes = states_includes
    
    @jax.named_scope("fbx.ForgetfulIterator")
    def __call__(
        self, 
        input: PyTree[Shaped[Array, "n_steps ..."]],  #! should have a batch dimension corresponding to time steps
        state: StateT,   # initial state
        key: PRNGKeyArray,
    ) -> StateT:  #! Adds a batch dimension, actually.
        key1, key2, key3 = jr.split(key, 3)
        
        init_input = tree_take(input, 0)
        states = self.init_arrays(init_input, state, key2)
        
        if os.environ.get('FEEDBAX_DEBUG', False) == "True": 
            for i in tqdm(range(self.n_steps),
                          desc="steps"):
                input, states, key3 = self._body_func(i, (input, states, key3))
                
            return states
                 
        _, states, _ = lax.fori_loop(
            0, 
            self.n_steps, 
            self._body_func,
            (input, states, key3),
        )
        
        return states

    @jax.named_scope("fbx.ForgetfulIterator._body_func")
    def _body_func(self, i: int, x: Tuple) -> Tuple:
        inputs, states, key = x
        
        key1, key2 = jr.split(key)
        
        # Since we optionally store the trajectories of only some of the states,
        # as specified by `states_includes`, we need to partition these out 
        # so we can index them, then recombine with the states for which only 
        # a single time step (the current one) is stored.
        states_mem, state_nomem = eqx.partition(states, self.states_includes)
        state_mem, input = tree_take((states_mem, inputs), i)
        state = eqx.combine(state_mem, state_nomem)
        
        state = self.step(input, state, key1)
        
        # Likewise, split the resulting states into those which are stored,
        # which are then assigned to the next index in the trajectory, and 
        # recombined with the single-timestep states.
        state_mem, state_nomem = eqx.partition(state, self.states_includes)        
        states_mem = tree_set(states_mem, state_mem, i + 1)
        states = eqx.combine(states_mem, state_nomem)
                
        return inputs, states, key2

    @jax.named_scope("fbx.ForgetfulIterator.init_arrays")
    def init_arrays(
        self, 
        input,  
        init_state: StateT,  
        key: PRNGKeyArray,
    ) -> StateT:  #! Adds a batch dimension re: `init_state`
        """Initialize the 
        
        The shape of the arrays in the PyTree(s) returned by `step` is
        automatically inferred, and used to initialize empty history arrays
        in which to store the states across all steps; `states_includes` can be
        used to specify which states to store. By default, all are stored.
        """
        # Get the shape of the state output by `self.step`
        outputs = eqx.filter_eval_shape(
            self.step, 
            input, 
            init_state, 
            key=key,
        )
        
        # Generate empty trajectories for mem states
        scalars, array_structs = eqx.partition(
            eqx.filter(outputs, self.states_includes), 
            eqx.is_array_like  # False for jax.ShapeDtypeStruct
        )
        asarrays = eqx.combine(
            jax.tree_map(jnp.asarray, scalars), 
            array_structs
        )
        states = jax.tree_map(
            lambda x: jnp.zeros((self.n_steps, *x.shape), dtype=x.dtype),
            asarrays,
        )
    
        # Insert the init state for mem states; combine with no mem state
        init_state_mem, init_state_nomem = eqx.partition(
            init_state, self.states_includes
        )
        states = eqx.combine(
            tree_set(states, init_state_mem, 0), 
            init_state_nomem
        )
        
        return states  



