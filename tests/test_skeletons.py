"""

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
import jax.numpy as jnp
from typing import TypeVar

import equinox as eqx
import jax
import jax.tree as jt
import jax.tree_util as jtu
import pytest

from feedbax.mechanics.skeleton import AbstractSkeleton, AbstractSkeletonState, PointMass, StateT
from feedbax.mechanics.skeleton.arm import TwoLinkArm, TwoLinkArmState
from feedbax.state import CartesianState


logger = logging.getLogger(__name__)


# TODO: examine TwoLinkArm forward consistency so we can make this lower!
ALL_CLOSE_ATOL = 1e-4



def forward_cycle(
    skeleton: AbstractSkeleton[StateT],
    skeleton_state: StateT
) -> StateT:
    """Forward kinematics followed by inverse kinematics."""
    return skeleton.inverse_kinematics(skeleton.effector(skeleton_state))


def inverse_cycle(
    skeleton: AbstractSkeleton[StateT],
    cartesian_state: CartesianState
) -> CartesianState:
    """Inverse kinematics followed by forward kinematics."""
    return skeleton.effector(skeleton.inverse_kinematics(cartesian_state))


TwoLinkArm = TwoLinkArm()


def test_TwoLinkArm_forward_consistency():

    skeleton_state = TwoLinkArmState()
    skeleton_state_cycle = forward_cycle(TwoLinkArm, skeleton_state)

    if not all(jtu.tree_leaves(jt.map(
            lambda x, y: jnp.allclose(x, y, atol=ALL_CLOSE_ATOL),
            skeleton_state,
            skeleton_state_cycle,
        ))):
        tree1_str = eqx.tree_pformat(skeleton_state, short_arrays=False)
        tree2_str = eqx.tree_pformat(skeleton_state_cycle, short_arrays=False)
        raise AssertionError(f"Not equal:\n\n{tree1_str}\n{tree2_str}")


def test_TwoLinkArm_inverse_consistency():

    effector_state = CartesianState(pos=jnp.array([0.0, 0.5]))
    effector_state_cycle = inverse_cycle(TwoLinkArm, effector_state)

    eqx.tree_pprint(effector_state, short_arrays=False)
    eqx.tree_pprint(effector_state_cycle, short_arrays=False)

    if not all(jtu.tree_leaves(jt.map(
            lambda x, y: jnp.allclose(x, y, atol=ALL_CLOSE_ATOL),
            effector_state,
            effector_state_cycle,
        ))):
        tree1_str = eqx.tree_pformat(effector_state, short_arrays=False)
        tree2_str = eqx.tree_pformat(effector_state_cycle, short_arrays=False)
        raise AssertionError(f"Not equal:\n\n{tree1_str}\n{tree2_str}")


pointmass = PointMass(mass=1.0)


def test_pointmass_forward_consistency():
    """
    These are identity functions for a point mass, so this should be trivial,
    but in the future we should test several points just to be sure.
    """
    skeleton_state = CartesianState(pos=jnp.array([0.0, 0.5]))
    skeleton_state_cycle = forward_cycle(pointmass, skeleton_state)

    assert all(jtu.tree_leaves(jt.map(
        lambda x, y: jnp.allclose(x, y),
        skeleton_state,
        skeleton_state_cycle,
    )))


def test_pointmass_inverse_consistency():

    effector_state = CartesianState(pos=jnp.array([0.0, 0.5]))
    effector_state_cycle = inverse_cycle(pointmass, effector_state)

    assert all(jtu.tree_leaves(jt.map(
        lambda x, y: jnp.allclose(x, y),
        effector_state,
        effector_state_cycle,
    )))


