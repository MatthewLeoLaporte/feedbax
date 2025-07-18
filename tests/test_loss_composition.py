"""

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

import logging

import equinox as eqx
import jax
import jax.tree as jt 


from feedbax.loss import (
    AbstractLoss,
    CompositeLoss, 
    EffectorFinalVelocityLoss,
    EffectorPositionLoss,
    NetworkActivityLoss,
    NetworkOutputLoss,
    power_discount,
)


logger = logging.getLogger(__name__)


def test_loss_composition():
    """Test that loss is constructed the same way via different methods."""
    loss_term_weights = dict(
        effector_position=1.,
        effector_final_velocity=1.,
        nn_output=1e-5,
        nn_hidden=1e-5,
    )

    loss_classes = dict(
        effector_position=EffectorPositionLoss,
        effector_final_velocity=EffectorFinalVelocityLoss,
        nn_output=NetworkOutputLoss,  
        nn_hidden=NetworkActivityLoss,
    )

    loss_from_dicts = CompositeLoss(
        jt.map(lambda cls: cls(), loss_classes),
        weights=loss_term_weights,
    )
    
    loss_from_sum = jax.tree_util.tree_reduce(
        lambda x, y: x + y,   
        jt.map(
            lambda w, cls: w * cls(),
            loss_term_weights,
            loss_classes,
        ),
        is_leaf=lambda x: isinstance(x, AbstractLoss),
    )

    assert loss_from_dicts == loss_from_sum