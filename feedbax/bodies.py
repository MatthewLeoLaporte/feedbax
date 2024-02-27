"""Compositions of mechanics, controllers, and channels into sensorimotor loops.

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import logging
from typing import Any, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.channel import Channel, ChannelSpec, ChannelState
from feedbax.intervene import AbstractIntervenor
from feedbax.model import MultiModel
from feedbax.mechanics import Mechanics, MechanicsState
from feedbax.nn import NetworkState
from feedbax._staged import AbstractStagedModel, ModelStage
from feedbax.state import AbstractState, StateBounds
from feedbax.task import AbstractTask
from feedbax._tree import tree_sum_n_features


logger = logging.getLogger(__name__)


class SimpleFeedbackState(AbstractState):
    """Type of state PyTree operated on by [`SimpleFeedback`][feedbax.bodies.SimpleFeedback] instances.

    Attributes:
        mechanics: The state PyTree for a `Mechanics` instance.
        net: The state PyTree for a staged neural network.
        feedback: A PyTree of state PyTrees for each feedback channel.
    """

    mechanics: "MechanicsState"
    net: "NetworkState"
    feedback: PyTree[ChannelState]


def _convert_feedback_spec(
    feedback_spec: Union[
        ChannelSpec, PyTree[ChannelSpec, "T"], PyTree[Mapping[str, Any], "T"]
    ]
) -> PyTree[ChannelSpec, "T"]:

    if not isinstance(feedback_spec, PyTree[ChannelSpec]):
        feedback_spec_flat, feedback_spec_def = eqx.tree_flatten_one_level(
            feedback_spec
        )

        if isinstance(feedback_spec_flat, PyTree[Mapping]):
            feedback_specs_flat = jax.tree_map(
                lambda spec: ChannelSpec(**spec),
                feedback_spec_flat,
                is_leaf=lambda x: isinstance(x, Mapping),
            )
            return jtu.tree_unflatten(
                feedback_spec_def,
                feedback_specs_flat,
            )
        else:
            return [ChannelSpec(**feedback_spec)]

    return feedback_spec


class SimpleFeedback(AbstractStagedModel[SimpleFeedbackState]):
    """Model of one step around a feedback loop between a neural network
    and a mechanical model.

    Attributes:
        net: The neural network that outputs commands for the mechanical model.
        mechanics: The discretized model of plant dynamics.
        feedback_channels: A PyTree of feedback channels which may be delayed
            and noisy.
    """

    net: eqx.Module  # TODO: should be stateful
    mechanics: "Mechanics"
    feedback_channels: PyTree[Channel]
    _feedback_specs: PyTree[ChannelSpec]
    intervenors: Mapping[str, Sequence[AbstractIntervenor]]

    def __init__(
        self,
        net: eqx.Module,
        mechanics: "Mechanics",
        feedback_spec: Union[
            PyTree[ChannelSpec], PyTree[Mapping[str, Any]]
        ] = ChannelSpec(
            where=lambda mechanics_state: mechanics_state.plant.skeleton,
        ),
        intervenors: Optional[
            Union[
                Sequence[AbstractIntervenor], Mapping[str, Sequence[AbstractIntervenor]]
            ]
        ] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """
        Arguments:
            net: The neural network that outputs commands for the mechanical model.
            mechanics: The discretized model of plant dynamics.
            feedback_spec: A PyTree of `ChannelSpec` instances,
            intervenors: [Intervenors][feedbax.intervene.AbstractIntervenor] to add
                to the model at construction time.
        """
        self.net = net
        self.mechanics = mechanics
        self.intervenors = self._get_intervenors_dict(intervenors)

        # If `feedback_spec` is given as a `PyTree[Mapping]`, convert to
        # `PyTree[ChannelSpec]`.
        #
        # Allow nesting of mappings to one level, to allow the user to provide
        # (say) a dict of dicts.
        feedback_specs = _convert_feedback_spec(feedback_spec)

        init_mechanics_state = mechanics.init()

        def _build_feedback_channel(spec: ChannelSpec):
            return Channel(spec.delay, spec.noise_std, jnp.nan).change_input(
                spec.where(init_mechanics_state)
            )

        self.feedback_channels = jax.tree_map(
            lambda spec: _build_feedback_channel(spec),
            feedback_specs,
            is_leaf=lambda x: isinstance(x, ChannelSpec),
        )
        self._feedback_specs = feedback_specs

    # def update_feedback(
    #     self,
    #     input: "MechanicsState",
    #     state: PyTree[ChannelState, 'T'],
    #     *,
    #     key: Optional[Array] = None
    # ) -> PyTree[ChannelState, 'T']:
    #     """Send current feedback states through channels, and return delayed feedback."""
    #     # TODO: separate keys for the different channels
    #     return jax.tree_map(
    #         lambda channel, spec, state: channel(spec.where(input), state, key=key),
    #         self.feedback_channels,
    #         self._feedback_specs,
    #         state,
    #         is_leaf=lambda x: isinstance(x, Channel),
    #     )

    @property
    def _feedback_module(self):
        return MultiModel(self.feedback_channels)

    @property
    def model_spec(self) -> OrderedDict[str, ModelStage]:
        """Specifies the stages of the model in terms of state operations."""
        return OrderedDict(
            {
                "update_feedback": ModelStage(
                    callable=lambda self: self._feedback_module,
                    where_input=lambda input, state: jax.tree_map(
                        lambda spec: spec.where(state.mechanics),
                        self._feedback_specs,
                        is_leaf=lambda x: isinstance(x, ChannelSpec),
                    ),
                    where_state=lambda state: state.feedback,
                ),
                "nn_step": ModelStage(
                    callable=lambda self: self.net,
                    where_input=lambda input, state: (
                        input,
                        # Get the output state for each feedback channel.
                        jax.tree_map(
                            lambda state: state.output,
                            state.feedback,
                            is_leaf=lambda x: isinstance(x, ChannelState),
                        ),
                    ),
                    where_state=lambda state: state.net,
                ),
                "mechanics_step": ModelStage(
                    callable=lambda self: self.mechanics,
                    where_input=lambda input, state: state.net.output,
                    where_state=lambda state: state.mechanics,
                ),
            }
        )

    def init(
        self,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """Return a default state for the model."""
        keys = jr.split(key, 3)

        return SimpleFeedbackState(
            mechanics=self.mechanics.init(key=keys[0]),
            net=self.net.init(key=keys[1]),
            feedback=self._feedback_module.init(key=keys[2]),
        )

    @property
    def memory_spec(self) -> SimpleFeedbackState:
        """Specifies which states should typically be remembered.

        For example, [`ForgetfulIterator`][feedbax.iterate.ForgetfulIterator]
        stores trajectories of states. However it doesn't usually make sense to
        store `states.feedback.queue` for every timestep, because it contains
        info that is already available if `states.mechanics` is stored at every
        timestep. If the feedback delay is 5 steps, `ForgetfulIterator` could
        end up with 5 extra copies of all the parts of `states.mechanics` that
        are part of the feedback. So it may be better not to store
        `states.feedback.queue`.

        This property will be used by `ForgetfulIterator`, but will be ignored
        by [`Iterator`][feedbax.iterate.Iterator], which remembers the full
        state indiscriminately—it's faster, but may use more memory.
        """
        return SimpleFeedbackState(
            mechanics=self.mechanics.memory_spec,
            net=self.net.memory_spec,
            feedback=jax.tree_map(
                lambda channel: channel.memory_spec,
                self.feedback_channels,
                is_leaf=lambda x: isinstance(x, eqx.Module),
            ),
        )

    @property
    def bounds(self) -> PyTree[StateBounds]:
        """Specifies the bounds of the state."""
        return SimpleFeedbackState(
            mechanics=self.mechanics.bounds,
            net=self.net.bounds,
            feedback=jax.tree_map(
                lambda channel: channel.bounds,
                self.feedback_channels,
                is_leaf=lambda x: isinstance(x, eqx.Module),
            ),
        )

    @staticmethod
    def get_nn_input_size(
        task: "AbstractTask",
        mechanics: "Mechanics",
        feedback_spec: Union[
            PyTree[ChannelSpec], PyTree[Mapping[str, Any]]
        ] = ChannelSpec(where=lambda mechanics_state: mechanics_state.plant.skeleton),
    ) -> int:
        """Determine how many scalar input features the neural network needs.

        This is a static method because its logic (number of network inputs =
        number of task inputs + number of feedback inputs from `mechanics`)
        is related to the structure of `SimpleFeedback`. However, it is
        not an instance method because we want to construct the network
        before we construct `SimpleFeedback`.
        """
        init_mechanics_state = mechanics.init()
        example_feedback = jax.tree_map(
            lambda spec: spec.where(init_mechanics_state),
            _convert_feedback_spec(feedback_spec),
            is_leaf=lambda x: isinstance(x, ChannelSpec),
        )
        n_feedback = tree_sum_n_features(example_feedback)
        example_trial_spec = task.get_train_trial(key=jr.PRNGKey(0))
        n_task_inputs = tree_sum_n_features(example_trial_spec.inputs)
        return n_feedback + n_task_inputs

    def state_consistency_update(
        self, state: SimpleFeedbackState
    ) -> SimpleFeedbackState:
        """Adjust the state

        Update the plant configuration state, given that the user has
        initialized the effector state.

        Also fill the feedback queues with the initial feedback states. This
        is less problematic than passing all zeros.

        TODO:
        - Check which of the two (effector or config) initialized, and update the other one.
          Might require initializing them to NaN or something in `init`.
        - Only initialize feedback channels whose *queues* are NaN, don't just check if
          the entire channel is NaN and updated all-or-none of them.
        """
        state = eqx.tree_at(
            lambda state: state.mechanics.plant.skeleton,
            state,
            self.mechanics.plant.skeleton.inverse_kinematics(state.mechanics.effector),
        )

        # If the PyTree of feedback channel states is full of NaNs, fill the channel queues
        # with the current values of the states to be fed back. By initializing `Channel` with
        # NaN and then performing this check/fill, we avoid passing zeros as feedback, which
        # is more incorrect.
        def _fill_feedback_queues(state):
            return eqx.tree_at(
                lambda state: state.feedback,
                state,
                jax.tree_map(
                    lambda channel_state, spec, channel: eqx.tree_at(
                        lambda channel_state: channel_state.queue,
                        channel_state,
                        channel.delay * (spec.where(state.mechanics),),
                    ),
                    state.feedback,
                    self._feedback_specs,
                    self.feedback_channels,
                    is_leaf=lambda x: isinstance(x, ChannelState),
                ),
            )

        feedback_state_isnan = jax.flatten_util.ravel_pytree(
            jax.tree_map(lambda x: jnp.isnan(x), state.feedback)
        )[0]

        state = jax.lax.cond(
            jnp.all(feedback_state_isnan),
            _fill_feedback_queues,
            lambda state: state,
            state,
        )

        return state
