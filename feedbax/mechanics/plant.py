"""

:copyright: Copyright 2023 by Matt L. Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

from abc import abstractmethod, abstractproperty
from collections import OrderedDict
import logging
from typing import Dict, Optional, Sequence, Tuple, TypeVar, Union

import equinox as eqx
from equinox import AbstractVar
import jax
from jax import Array
import jax.numpy as jnp
from jaxtyping import Float
from feedbax.intervene import AbstractIntervenor
from feedbax.mechanics.muscle import AbstractMuscle, AbstractMuscleState
from feedbax.mechanics.skeleton.arm import TwoLink
from feedbax.mechanics.skeleton.skeleton import AbstractSkeleton

from feedbax.model import AbstractModel, AbstractModelState


logger = logging.getLogger(__name__)


class AbstractSkeletonState(AbstractModelState):
    ...
    
    
class PlantState(AbstractModelState):
    skeleton: AbstractSkeletonState
    muscles: AbstractMuscleState


class AbstractPlant(AbstractModel[PlantState]):
    
    skeleton: AbstractVar[AbstractSkeleton]
    muscle_model: AbstractVar[AbstractMuscle]
    
    @abstractproperty
    def model_spec(self):
        ...
    
    @abstractproperty 
    def dynamics_spec(self):
        ... 
    
    @abstractproperty 
    def memory_spec(self):
        ...
    
    @abstractmethod
    def init(self) -> PlantState:
        ...
        
    @abstractproperty 
    def input_size(self) -> int:
        ...


class MuscledArm(AbstractPlant):
    """"""
    skeleton: AbstractSkeleton 
    muscle_model: AbstractMuscle
    activator: eqx.Module
    
    n_muscles: int
    moment_arms: Float[Array, "links=2 muscles"]
    theta0: Float[Array, "links=2 muscles"] 
    l0: Float[Array, "muscles"] 
    f0: Float[Array, "muscles"] 
    
    intervenors: Dict[str, AbstractIntervenor] 
    
    def __init__(
        self, 
        muscle_model, 
        activator,
        skeleton=TwoLink(), 
        moment_arms=jnp.array(((2.0, -2.0, 0.0, 0.0, 1.50, -2.0),  # [cm]
                               (0.0, 0.0, 2.0, -2.0, 2.0, -1.50))),
        theta0=2 * jnp.pi * jnp.array(((15.0, 4.88, 0.0, 0.0, 4.5, 2.12), # [rad]
                                       (0.0, 0.0, 80.86, 109.32, 92.96, 91.52))) / 360.,  
        l0=jnp.array((7.32, 3.26, 6.4, 4.26, 5.95, 4.04)),  # [cm]
        f0=1., #31.8 * jnp.array((22., 12., 18., 14., 5., 10.)),  # [N] = [N/cm^2] * [cm^2]
        intervenors: Optional[Union[Sequence[AbstractIntervenor],
                                    Dict[str, Sequence[AbstractIntervenor]]]] \
            = None,
        *,
        key: Optional[jax.Array] = None,
    ):
        self.skeleton = skeleton
        self.muscle_model = muscle_model
        self.activator = activator
        
        if not theta0.shape[1] == l0.shape[0] == moment_arms.shape[1]:
            raise ValueError(
                "moment_arms, theta0, and l0 must have the same number of "
                "columns (i.e. number of muscles)"
            )
        
        self.moment_arms = moment_arms  
        self.theta0 = theta0
        self.l0 = l0
        self.f0 = f0
        self.n_muscles = moment_arms.shape[1]
        
        self.intervenors = self._get_intervenors_dict(intervenors)        
    
    def dynamics_spec(self):
        return dict({
            "muscle_activation": (
                lambda self: self.activator,  # vector field
                lambda input, state: input,  # system input
                lambda state: state.muscles.activation,  # system state
            ),
            "skeleton": (
                lambda self: self.skeleton,
                lambda input, state: state.skeleton.torque,
                lambda state: state.skeleton,
            ),
        })
    
    @property
    def model_spec(self):
        return OrderedDict({
            "muscle_update": (
                lambda self: self.muscle_update,
                lambda input, state: (input, state.skeleton),
                lambda state: (
                    state.muscles.length,
                    state.muscles.velocity,
                    state.muscles.tension,
                ),
            ),
            "muscle_torques": (
                lambda self: self.muscle_torques,
                lambda input, state: state.muscles,
                lambda state: state.skeleton.torque,
            ),
        })
    
    def muscle_update(
        self, 
        input: Tuple[Array, AbstractSkeletonState], 
        state, 
        *, 
        key=None
    ):
        muscle_input, skeleton_state = input
        length = self._muscle_length(skeleton_state.theta)
        velocity = self._muscle_velocity(skeleton_state.d_theta)
        tension = self.muscle_model(length, velocity, muscle_input)
        
        return (length, velocity, tension)

    def _muscle_length(self, theta):
        moment_arms, l0, theta0 = self.theta0, self.l0, self.theta0
        l = 1 + (moment_arms[0] * (theta0[0] - theta[0]) + moment_arms[1] * (theta0[1] - theta[1])) / l0
        return l
    
    def _muscle_velocity(self, d_theta):
        moment_arms, l0 = self.theta0, self.l0
        v = (moment_arms[0] * d_theta[0] + moment_arms[1] * d_theta[1]) / l0
        return v

    def muscle_torques(self, input, state, *, key=None):
        torque = self.moment_arms @ (self.f0 * input.tension)
        return torque
        
    @property
    def memory_spec(self):
        return PlantState(
            skeleton=True,
            muscles=True,
        )
    
    def init(self, skeleton=None) -> PlantState:
        if skeleton is None:
            skeleton = self.skeleton.init()
        
        return PlantState(
            skeleton=skeleton,
            muscles=self.muscle_model.init(self.n_muscles),
        )
    
    @property
    def input_size(self):
        return self.n_muscles