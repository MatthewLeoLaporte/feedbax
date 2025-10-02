"""Pre-built loss functions for common tasks.

:copyright: Copyright 2024 by Matt L. Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Optional

import jax.numpy as jnp

from feedbax.loss import (
    CompositeLoss,
    EffectorFinalVelocityLoss,
    EffectorFixationLoss,
    EffectorPositionLoss,
    EffectorVelocityLoss,
    NetworkActivityLoss,
    NetworkOutputLoss,
    TargetSpec,
    TargetStateLoss,
    power_discount,
    target_final_state,
    target_zero,
)

logger = logging.getLogger(__name__)


def simple_reach_loss(
    effector_position: float = 1.0,
    effector_final_velocity: float = 1.0,
    nn_output: float = 1e-5,
    nn_hidden: float = 1e-5,
) -> CompositeLoss:
    """A typical loss function for a simple reaching task.

    Arguments:
        loss_term_weights: Maps loss term names to term weights. If `None`,
            a typical set of default weights is used.
        discount_exp: The exponent of the power function used to discount
            the position error, back in time from the end of trials. Larger
            values lead to penalties that are more concentrated at the end
            of trials. If zero, all time steps are weighted equally.
    """
    return CompositeLoss(
        dict(
            effector_position=TargetStateLoss(
                "Effector position",
                where=lambda state: state.mechanics.effector.pos,
                norm=lambda x: jnp.sum(x**2, axis=-1),
                # norm=lambda *args, **kwargs: (
                #     # Euclidean distance
                #     jnp.linalg.norm(*args, axis=-1, **kwargs) ** 2
                # ),
            ),
            effector_final_velocity=TargetStateLoss(
                "Effector final velocity",
                where=lambda state: state.mechanics.effector.vel,
                # By indexing out the final timestep only, this loss must
                # be paired with an `AbstractTask` that supplies a
                # single-timestep target value.
                spec=target_zero & target_final_state,
            ),
            nn_output=TargetStateLoss(
                "Command",
                where=lambda state: state.efferent.output,
                spec=target_zero,
            ),
            nn_hidden=TargetStateLoss(
                "NN activity",
                where=lambda state: state.net.hidden,
                spec=target_zero,
            ),
        ),
        weights=dict(
            effector_position=effector_position,
            effector_final_velocity=effector_final_velocity,
            nn_output=nn_output,
            nn_hidden=nn_hidden,
        ),
    )


def delayed_reach_loss(
    effector_fixation: float = 1.0,
    effector_position: float = 1.0,
    effector_final_velocity: float = 1.0,
    nn_output: float = 1e-4,
    nn_hidden: float = 1e-5,
) -> CompositeLoss:
    """A typical loss function for a `DelayedReaches` task."""
    return CompositeLoss(
        dict(
            # these assume a particular PyTree structure to the states returned by the model
            # which is why we simply instantiate them
            effector_fixation=EffectorFixationLoss(),
            effector_position=TargetStateLoss(
                "Effector position",
                where=lambda state: state.mechanics.effector.pos,
                norm=lambda x: jnp.sum(x**2, axis=-1),
                # norm=lambda *args, **kwargs: (
                #     # Euclidean distance
                #     jnp.linalg.norm(*args, axis=-1, **kwargs) ** 2
                # ),
            ),
            effector_final_velocity=TargetStateLoss(
                "Effector final velocity",
                where=lambda state: state.mechanics.effector.vel,
                # By indexing out the final timestep only, this loss must
                # be paired with an `AbstractTask` that supplies a
                # single-timestep target value.
                spec=target_zero & target_final_state,
            ),
            nn_output=TargetStateLoss(
                "Command",
                where=lambda state: state.efferent.output,
                spec=target_zero,
            ),
            nn_hidden=TargetStateLoss(
                "NN activity",
                where=lambda state: state.net.hidden,
                spec=target_zero,
            ),
        ),
        weights=dict(
            effector_fixation=effector_fixation,
            effector_position=effector_position,
            effector_final_velocity=effector_final_velocity,
            nn_output=nn_output,
            nn_hidden=nn_hidden,
        ),
    )


def hold_loss(
    loss_term_weights: Optional[Mapping[str, float]] = None,
) -> CompositeLoss:
    """A typical loss function for a postural stabilization task.

    Arguments:
        loss_term_weights: Maps loss term names to term weights. If `None`,
            a typical set of default weights is used.
    """
    if loss_term_weights is None:
        loss_term_weights = dict(
            effector_position=1.0,
            effector_velocity=1e-5,
            nn_output=1e-5,
            nn_hidden=1e-5,
        )
    return CompositeLoss(
        dict(
            effector_position=TargetStateLoss(
                "Effector position",
                where=lambda state: state.mechanics.effector.pos,
                # Euclidean distance
                norm=lambda x: jnp.sum(x**2, axis=-1),
                # norm=lambda *args, **kwargs: (
                #     jnp.linalg.norm(*args, axis=-1, **kwargs) ** 2
                # ),
            ),
            effector_velocity=TargetStateLoss(
                "Effector velocity",
                where=lambda state: state.mechanics.effector.vel,
                spec=target_zero,
            ),
            nn_output=TargetStateLoss(
                "Command",
                where=lambda state: state.efferent.output,
                spec=target_zero,
            ),
            nn_hidden=TargetStateLoss(
                "NN activity",
                where=lambda state: state.net.hidden,
                spec=target_zero,
            ),
        ),
        weights=loss_term_weights,
    )
