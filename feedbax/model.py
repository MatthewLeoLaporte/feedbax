"""Modules that compose other modules into control contexts.

For example, classes defined here could be used to to model a sensorimotor 
loop, a body, or (perhaps) multiple interacting bodies. 

TODO:
- Maybe this should be renamed... mainly because it might be confused with 
`AbstractTask` in terms of its purpose.

:copyright: Copyright 2023 by Matt L. Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
from collections import OrderedDict
import copy
from functools import cached_property, wraps
import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable, 
    Dict, 
    Generic, 
    Optional, 
    Sequence, 
    Tuple,
    Type, 
    TypeVar, 
    Union,
)

import equinox as eqx
from equinox import AbstractVar
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree
import numpy as np

from feedbax.channel import Channel, ChannelState
from feedbax.intervene import AbstractIntervenor
if TYPE_CHECKING:
    from feedbax.mechanics import Mechanics, MechanicsState
from feedbax.task import AbstractTask, AbstractTaskInput
from feedbax.utils import tree_sum_n_features, tree_pformat_indent

if TYPE_CHECKING:
    from feedbax.networks import NetworkState


logger = logging.getLogger(__name__)


N_DIM = 2


class AbstractModelState(eqx.Module):
    ...
    

StateT = TypeVar("StateT", bound=AbstractModelState)


class AbstractModel(eqx.nn.StatefulLayer, Generic[StateT]):
    """
    
    TODO:
    - Should this be a generic of `StateT | Array`, or something?
      - i.e. what about models that return a single array or some other
        kind of PyTree? As in linear regression, PCA...
    """
    
    @abstractmethod
    def __call__(
        self,
        input,
        state: StateT, 
        key: jax.Array,
    ) -> StateT:
        """Update the model state given inputs and a prior state."""
        ...

    @abstractproperty
    def step(self) -> "AbstractModel[StateT]":
        """Interface to a single model step.
        
        For non-iterated models, this should trivially return `step`.
        """
        ...
    
    def state_consistency_update(
        self, 
        state: StateT,
    ) -> StateT:
        """Make sure the model state is self-consistent.
        
        The default behaviour is to just return the same state. However, in
        models where it is customary to initialize (say) the effector state 
        but not the plant configuration state, this method should be used 
        to ensure that the configuration state is initialized properly. 
        
        This avoids having to define how both states should be initialized
        in `AbstractTask`, which would require knowledge not only of the 
        structure of the state, but also of the model interface that 
        provides particular operations for modifying the state. Though, I'm not 
        sure that would be a bad thing, as the model state objects tend to 
        mirror the operations done on them anyway.
        
        However, this approach also has the advantage that these consistency
        checks can be arbitrarily complex and important to the model 
        functioning properly; it is useful for the model to be able to 
        render a state consistent, with respect to its other normal operations.
        """
        return state
    
    @abstractmethod
    def init(
        self,
        *,
        key: Optional[jax.Array] = None,
    ) -> StateT:
        """Return an initial state for the model."""
        ...
    

class AbstractStagedModel(AbstractModel[StateT]):
    """Base class for state-dependent models.
    
    To define a new model, the following should be implemented:
    
        1. A concrete subclass of `AbstractModelState` that defines the PyTree
           structure of the full model state. The fields may be types of 
           `AbstractModelState` as well, in the case of nested models.
        2. A concrete subclass of this class with:
            i. the appropriate components for doing state transformations;
            either `eqx.Module`/`AbstractModel`-typed fields, or instance 
            methods, each with the signature `input, state, *, key`;
            ii. a `model_spec` property that specifies a series of named 
            operations on parts of the full model state, using those component 
            modules; and
            iii. an `init` method that returns an initial model state.
            
    The motivation behind the level of abstraction of this class is that it
    gives a name to each functional stage of a composite model, so that after 
    model instantiation or training, the user can choose to insert additional 
    state interventions to be executed prior to any of the named stages. 
    """
    intervenors: AbstractVar[Dict[str, Sequence[AbstractIntervenor]]]
    
    def __call__(
        self, 
        input, # AbstractTaskInput 
        state: StateT, 
        key: jax.Array,
    ) -> StateT:
        
        with jax.named_scope(type(self).__name__):
            
            keys = jr.split(key, len(self._stages))
            
            for label, key in zip(self._stages, keys):
                intervenors, module, module_input, substate_where = \
                    self._stages[label]
                
                key1, key2 = jr.split(key)
                
                if os.environ.get('FEEDBAX_DEBUG', False) == "True": 
                    debug_strs = [tree_pformat_indent(x, indent=4) for x in 
                        (
                            module(self),
                            module_input(input, state),
                            substate_where(state),
                        )
                    ]
                    
                    logger.debug(
                        f"Module: {type(self).__name__} \n"
                        f"Stage: {label} \n"
                        f"Stage module:\n{debug_strs[0]}\n"
                        f"Input:\n{debug_strs[1]}\n"
                        f"Substate:\n{debug_strs[2]}\n"
                    )                  
                
                for intervenor in intervenors:
                    # TODO: Whether this modifies the state depends on `input`.
                    state = intervenor(input, state, key=key1)
                
                state = eqx.tree_at(
                    substate_where, 
                    state, 
                    module(self)(
                        module_input(input, state), 
                        substate_where(state), 
                        key=key2,
                    ),
                )
        
        return state

    @abstractproperty
    def model_spec(self) -> OrderedDict[str, Tuple[
        eqx.Module,
        Callable[[StateT], PyTree],
        Sequence[Callable[[AbstractTaskInput, StateT], PyTree]],
    ]]:
        """Specifies the model in terms of substate operations.
        
        Each tuple in the dict has the following elements:
        
            1. A callable (e.g. an `eqx.Module` or method) that takes `input`
               and a substate of the model state and returns an updated 
               substate;
            2. A function that selects the parts of the model input and state 
               PyTrees that are passed to the module, i.e. its `input`; and
            3. A function that selects the part of the model state PyTree to 
               be updated, i.e. the part of the model state passed as the
               second (`state`) argument to the callable, and updated by the
               PyTree returned by the callable.
               
        It's necessary to use `OrderedDict` because `jax.tree_util` still
        sorts `dict` keys, and we can't have that.
        """
        ...
  
    @cached_property 
    def _stages(self) -> Tuple[Tuple[
        Sequence[AbstractIntervenor],
        eqx.Module,
        Callable[[StateT], PyTree],
        Sequence[Callable[[AbstractTaskInput, StateT], PyTree]],
    ]]: 
        """Zips up the user-defined intervenors with `model_spec`."""
        #! should not be referred to in `__init__`, at least before defining `intervenors`
        return jax.tree_map(
            lambda x, y: (y,) + x, 
            self.model_spec, 
            jax.tree_map(tuple, self.intervenors, 
                        is_leaf=lambda x: isinstance(x, list)),
            is_leaf=lambda x: isinstance(x, tuple),
        )
    
    def _get_intervenors_dict(
        self, intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                          Dict[str, Sequence[AbstractIntervenor]]]]
    ):
        intervenors_dict = jax.tree_map(
            lambda _: [], 
            self.model_spec,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        
        if intervenors is not None:
            if isinstance(intervenors, Sequence):
                # By default, place interventions in the first stage.
                intervenors_dict.update({'get_feedback': list(intervenors)})
            elif isinstance(intervenors, dict):
                intervenors_dict.update(
                    jax.tree_map(list, intervenors, 
                                 is_leaf=lambda x: isinstance(x, Sequence)))
            else:
                raise ValueError("intervenors not a sequence or dict of sequences")
        
        return intervenors_dict
    
    @abstractproperty
    def memory_spec(self) -> PyTree[bool]:
        """Specifies which states should typically be remembered by callers."""
        ...
        
    @property 
    def step(self):
        """This assumes all staged models will specify single update steps,
        not iterated models.
        
        This allows classes like `AbstractTask` and `TaskTrainer` to refer
        to methods of a single step of the model, without having to know whether
        the model step is wrapped in an iterator; `AbstractIterator` concretes will 
        return `self._step` instead.
        """
        return self 
        

class SimpleFeedbackState(AbstractModelState):
    mechanics: "MechanicsState"
    network: "NetworkState"
    feedback: ChannelState


class SimpleFeedback(AbstractStagedModel[SimpleFeedbackState]):
    """Simple feedback loop with a single RNN and single mechanical system.
    
    TODO:
    - PyTree of force inputs (in addition to control inputs) to mechanics
    - Could insert intervenors as stages in the model, but I think it makes 
    more sense for modules to be designed-in, which then provides the user 
    with labels for stages at which interventions can be applied, resulting 
    in a dict of interventions that mirrors the dict of stages.
    """
    net: eqx.Module  
    mechanics: "Mechanics"
    feedback_channel: Channel
    delay: int 
    feedback_leaves_func: Callable[[PyTree], PyTree]  
    
    intervenors: Dict[str, Sequence[AbstractIntervenor]]
    
    def __init__(
        self, 
        net: eqx.Module, 
        mechanics: "Mechanics", 
        delay: int = 0, 
        feedback_leaves_func: Callable[["MechanicsState"], PyTree] \
            = lambda mechanics_state: mechanics_state.plant.skeleton,
        feedback_noise_std: Optional[float] = None,
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Dict[str, Sequence[AbstractIntervenor]]]] \
            = None,
        *,
        key: Optional[jax.Array] = None,
    ):
        self.net = net
        self.mechanics = mechanics
        self.feedback_leaves_func = feedback_leaves_func
        self.delay = delay + 1  # indexing: delay=0 -> storage len=1
        self.feedback_channel = Channel(delay, feedback_noise_std).change_input(
            feedback_leaves_func(mechanics.init())
        )
        self.intervenors = self._get_intervenors_dict(intervenors)
        
    @property
    def model_spec(self):
        """Specifies the stages of the model in terms of state operations.
        
        Note that the module has to be given as a function that obtains it from `self`, 
        otherwise it won't use modules that have been updated during training.
        I'm not sure why the references are kept across model updates...
        """
        return OrderedDict({
            'get_feedback': (
                lambda self: self.feedback_channel,  # module that operates on it
                lambda _, state: self.feedback_leaves_func(state.mechanics),  # inputs
                lambda state: state.feedback,  # substate it operates on
            ),
            'nn_step': (
                lambda self: self.net,
                lambda input, state: (input, state.feedback.output),
                lambda state: state.network,                
            ),
            'mechanics_step': (
                lambda self: self.mechanics,
                lambda _, state: state.network.output,
                lambda state: state.mechanics,
            ),
        })       

    def init(
        self, 
        *,
        key: Optional[jax.Array] = None,
    ): 
        """Return an initial state for the model.
        
        TODO:
        - We can probably use `eval_shape` more generally when an init is not 
          available, which may be the case for simpler modules. The 
          `model_spec` could be used to determine the relevant info flow.
        """
        mechanics_state = self.mechanics.init()

        return SimpleFeedbackState(
            mechanics=mechanics_state,
            network=self.net.init(),
            feedback=self.feedback_channel.init(),
        )
    
    @property
    def memory_spec(self) -> SimpleFeedbackState:
        """Specifies which states should typically be remembered by callers.
        
        For example, `fbx.Iterator` stores trajectories of states, however it
        doesn't usually make sense to store `states.feedback.queue` for every
        timestep, because it contains info that is already available to
        `Iterator` if `states.mechanics` is stored at every timestep. If the
        feedback delay is 5 steps, `Iterator` would otherwise end up with 5 
        extra copies of all the parts of `states.mechanics` that are part of 
        the feedback.
        
        In particular, this information will be used by `Iterator`, but will
        be ignored by `SimpleIterator`, which remembers the full state 
        indiscriminately to favour speed over memory usage.
        
        NOTE: It makes sense for this to be here since it has to do with the
        logic of the feedback loop, i.e. that queue is just transient internal 
        memory of another variable in the loop. 
        """
        return SimpleFeedbackState(
            mechanics=self.mechanics.memory_spec, 
            network=self.net.memory_spec,
            feedback=ChannelState(output=True, queue=False)
        )

    @staticmethod
    def get_nn_input_size(
        task: AbstractTask, 
        mechanics: "Mechanics", 
        feedback_leaves_func=lambda mechanics_state: mechanics_state.plant.skeleton,
    ) -> int:
        """Determine how many scalar input features the neural network needs.
        
        This is a static method because its logic (number of network inputs =
        number of task inputs + number of feedback inputs from `mechanics`) 
        is related to the structure of `SimpleFeedback`. However, it is 
        not an instance method because we want to construct the network
        before we construct `SimpleFeedback`.
        """
        example_feedback = feedback_leaves_func(mechanics.init())
        n_feedback = tree_sum_n_features(example_feedback)
        example_trial_spec = task.get_train_trial(key=jr.PRNGKey(0))[0]
        n_task_inputs = tree_sum_n_features(example_trial_spec.input)
        return n_feedback + n_task_inputs
    
    def state_consistency_update(
        self, 
        state: SimpleFeedbackState
    ) -> SimpleFeedbackState:
        """Update the plant configuration state, given that the user has 
        initialized the effector state.
        
        TODO: 
        - Check which of the two is initialized, and update the other one.
          Might require initializing them to NaN or something in `init`.
        """
        return eqx.tree_at(
            lambda state: state.mechanics.plant.skeleton,
            state, 
            jax.vmap(self.mechanics.plant.skeleton.inverse_kinematics)(
                state.mechanics.effector
            ),
        )
    

def add_intervenor(
    model: AbstractStagedModel[StateT], 
    intervenor: AbstractIntervenor,
    stage_name: Optional[str] = None,
    **kwargs
) -> AbstractStagedModel[StateT]:
    """Return an updated model with an added intervenor.
    
    This is just a convenience for passing a single intervenor to `add_intervenor`.
    """
    if stage_name is not None:
        return add_intervenors(model, {stage_name: [intervenor]}, **kwargs)
    else:
        return add_intervenors(model, [intervenor], **kwargs)


def add_intervenors(
    model: AbstractStagedModel[StateT], 
    intervenors: Union[Sequence[AbstractIntervenor],
                       Dict[str, Sequence[AbstractIntervenor]]], 
    where: Callable[[AbstractStagedModel[StateT]], Any] = lambda model: model,
    keep_existing: bool = True,
    *,
    key: Optional[jax.Array] = None
) -> AbstractStagedModel[StateT]:
    """Return an updated model with added intervenors.
    
    Intervenors are added to `where(model)`, which by default is 
    just `model` itself.
    
    If intervenors are passed as a sequence, they are added to the first stage
    specified in `where(model).model_spec`, otherwise they should be passed in
    a dict where the keys refer to particular stages in the model spec.
    
    TODO:
    - Could this be generalized to `AbstractModel[StateT]`?
    """
    if keep_existing:
        existing_intervenors = where(model).intervenors
        
        if isinstance(intervenors, Sequence):
            # If a sequence is given, append to the first stage.
            first_stage_label = next(iter(existing_intervenors))
            intervenors_dict = eqx.tree_at(
                lambda intervenors: intervenors[first_stage_label],
                existing_intervenors,
                existing_intervenors[first_stage_label] + list(intervenors),
            )
        elif isinstance(intervenors, dict):
            intervenors_dict = copy.deepcopy(existing_intervenors)
            for label, new_intervenors in intervenors.items():
                intervenors_dict[label] += list(new_intervenors)
        else:
            raise ValueError("intervenors not a sequence or dict of sequences")

    for k in intervenors_dict:
        if k not in where(model).model_spec:
            raise ValueError(f"{k} is not a valid model stage for intervention")

    return eqx.tree_at(
        lambda model: where(model).intervenors,
        model, 
        intervenors_dict,
    )

def remove_intervenors(
    model: SimpleFeedback
) -> SimpleFeedback:
    """Return a model with no intervenors."""
    return add_intervenors(model, intervenors=(), keep_existing=False)


def wrap_stateless_module(module: eqx.Module):
    """Makes a 'stateless' module trivially compatible with state-passing.
    
    `AbstractModel` defines everything in terms of transformations of parts of
    a state PyTree. In each case, the substate that is operated on is passed
    to the module that returns the updated substate. However, in some cases
    the new substate does not depend on the previous substate. For example,
    a linear network layer takes some inputs and returns some outputs, and
    on the next iteration, the linear layer's outputs do not conventionally 
    depend on its previous outputs, like an RNN cell's would.
    """
    @wraps(module)
    def wrapped(input, state, *args, **kwargs):
        return module(input, *args, **kwargs)
    
    return wrapped