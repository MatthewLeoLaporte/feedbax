"""


:copyright: Copyright 2023-2024 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

from functools import cached_property
import logging
from typing import Optional

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray, Scalar
import numpy as np

from feedbax.mechanics.skeleton import AbstractSkeleton, AbstractSkeletonState
from feedbax.state import CartesianState, StateBounds
from feedbax.misc import SINCOS_GRAD_SIGNS, corners_2d


logger = logging.getLogger(__name__)


class TwoLinkArmState(AbstractSkeletonState):
    """The configuration state of a 2D arm with two rotational joints.

    Attributes:
        angle: The joint angles.
        d_angle: The angular velocities.
        torque: The torques on the joints.
    """

    angle: Float[Array, "... links=2"] = field(default_factory=lambda: jnp.zeros(2))
    d_angle: Float[Array, "... links=2"] = field(default_factory=lambda: jnp.zeros(2))
    torque: Float[Array, "... links=2"] = field(default_factory=lambda: jnp.zeros(2))


class TwoLinkArm(AbstractSkeleton[TwoLinkArmState]):
    """Model of a 2D arm with two straight rigid segments, and two rotational joints.

    Attributes:
        l: The lengths of the arm segments. Units: $[\\mathrm{L}]$.
        m: The masses of the segments. Units: $[\\mathrm{M}]$.
        I: The moments of inertia of the segments.
            Units: $[\\mathrm{M}\\cdot \\mathrm{L}^2]$.
        s: The distance from the joint center to the segment center of mass
            for each segment. Units: $[\\mathrm{L}]$.
        B: The joint friction matrix.
            Units: $[\\mathrm{M}\\cdot \\mathrm{L}^2\\cdot \\mathrm{T}^{-1}]$.
    """

    l: Float[Array, "links=2"] = field(converter=jnp.asarray)
    m: Float[Array, "links=2"] = field(converter=jnp.asarray)
    I: Float[Array, "links=2"] = field(converter=jnp.asarray)
    s: Float[Array, "links=2"] = field(converter=jnp.asarray)
    B: Float[Array, "2 2"] = field(converter=jnp.asarray)
    inertia_gain: float

    def __init__(
        self,
        l=(0.30, 0.33),  # [m]
        m=(1.4, 1.0),  # [kg]
        I=(0.025, 0.045),  # [kg m^2]
        s=(0.11, 0.16),  # [m]
        B=((0.05, 0.025), (0.025, 0.05)),  # [kg m^2 s^-1]
    ):
        self.l = l
        self.m = m
        self.I = I
        self.s = s
        self.B = B
        self.inertia_gain = 1.0

        #! initialize cached properties used by JAX operations
        # otherwise their initialization is a side effect
        self._a

    @jax.named_scope("fbx.TwoLinkArm")
    def vector_field(
        self,
        t: Scalar,
        state: TwoLinkArmState,
        input: Array,
    ) -> TwoLinkArmState:
        """Return the time derivatives of the arm's configuration state.

        Arguments:
            t: The time. Not used.
            state: The current state of the arm.
            input: The torques on the joints.
        """
        angle, d_angle = state.angle, state.d_angle

        # centripetal and coriolis torques
        c_vec = (
            jnp.array((-d_angle[1] * (2 * d_angle[0] + d_angle[1]), d_angle[0] ** 2))
            * self._a[1]
            * jnp.sin(angle[1])
        )

        # inertia matrix that maps torques -> angular accelerations
        cs1 = jnp.cos(angle[1])
        tmp = self._a[2] + self._a[1] * cs1
        inertia_mat = jnp.array(
            (
                (self._a[0] + 2 * self._a[1] * cs1, tmp),
                (tmp, self._a[2] * jnp.ones_like(cs1)),
            )
        )

        # Normally, `state.torque` only contains the torque induced by
        # linear force on the effector, as determined by inverse kinematics
        # during the "convert_effector_force" stage of `Mechanics`,
        # whereas `input` is the torque from the controller/muscles.
        net_torque = state.torque + input - c_vec.T - jnp.matmul(d_angle, self.B.T)

        dd_angle = jnp.linalg.inv(inertia_mat) @ net_torque

        return TwoLinkArmState(d_angle, dd_angle)

    def init(
        self,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> TwoLinkArmState:
        """Return a default state for the arm."""
        return TwoLinkArmState()

    @cached_property
    def _a(self):
        # this is a cached_property to avoid polluting the module's fields with private attributes
        return (
            # + m[1]*s[1]**2 + m[0]*s[0]**2
            self.I[0] + self.I[1] + self.m[1] * self.l[0] ** 2,
            #
            self.m[1] * self.l[0] * self.s[1],
            # + m[1] * s[1] ** 2
            self.I[1],
        )

    @cached_property
    def _lsqpm(self):
        lsq = self.l**2
        return (lsq[0] - lsq[1], lsq[0] + lsq[1])

    @property
    def input_size(self) -> int:
        """Number of input variables—in this case, joint torques."""
        return self.n_links  # per-joint torque

    # @property
    # def state_size(self) -> int:
    #     """Number of configuration state variables."""
    #     return self.n_links * 2  # per-joint angle and angular velocity

    @property
    def n_links(self) -> int:
        """Number of arm segments."""
        return 2

    @jax.named_scope("fbx.TwoLinkArm.inverse_kinematics")
    def inverse_kinematics(self, effector_state: CartesianState) -> TwoLinkArmState:
        """Return the configuration state of the arm, given the Cartesian state
        of the end effector.

        !!! Note ""
            There are two possible solutions to the inverse kinematics problem,
            for a two-link arm. This method returns the "elbow down" or
            "righty" solution.

        Arguments:
            effector_state: The state of the end effector.
        """
        pos = effector_state.pos
        l, lsqpm = self.l, self._lsqpm
        dsq = jnp.sum(pos**2)
        # No solution exists if `dsq` is outside of:
        #   `(l[0] - l[1], l[0] + l[1])`

        alpha = jnp.arccos((lsqpm[0] + dsq) / (2 * l[0] * jnp.sqrt(dsq)))
        gamma = jnp.arctan2(pos[1], pos[0])
        theta0 = gamma - alpha

        beta = jnp.arccos((lsqpm[1] - dsq) / (2 * l[0] * l[1]))
        theta1 = np.pi - beta
        # The "elbow up" or "lefty" solution is given by:
        #   `theta0 = gamma + alpha`
        #   `theta1 = beta - pi`

        angle = jnp.stack([theta0, theta1], axis=-1)

        # TODO: Don't calculate Jacobian twice, for `d_angle` and `torque`.
        # See https://robotics.stackexchange.com/a/6393
        d_angle = jnp.linalg.inv(self.effector_jac(angle)) @ effector_state.vel

        if effector_state.force is not None:
            torque = self.effector_force_to_torques(angle, effector_state.force)
        else:
            torque = None

        return TwoLinkArmState(angle=angle, d_angle=d_angle, torque=torque)

    def update_state_given_effector_force(
        self,
        effector_force: Array,
        state: TwoLinkArmState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> TwoLinkArmState:
        """Adds torques implied by a force on the end effector, to a
        configuration state of the arm.

        Arguments:
            effector_force: The force on the end effector.
            state: The configuration state of the arm to which to add
                the inferred torques.
            key: Unused. To satisfy the signature expected by
                `AbstractSkeleton`.
        """
        torque = self.effector_force_to_torques(
            state.angle,
            effector_force,
        )

        return eqx.tree_at(lambda state: state.torque, state, torque + state.torque)

    def effector_force_to_torques(
        self,
        angle: Float[Array, "links=2"],
        effector_force: Float[Array, "ndim=2"],
    ) -> Float[Array, "links=2"]:
        """Return the joint torques induced by a force on the end effector.

        Arguments:
            angle: The joint angles.
            effector_force: The force on the end effector.
        """
        torque = self.effector_jac(angle).T @ effector_force
        return torque

    def effector_jac(
        self, angle: Float[Array, "links=2"]
    ) -> Float[Array, "ndim=2 links=2"]:
        """Return the Jacobian of end effector position with respect to joint
        angles.

        Arguments:
            angle: The joint angles.
        """
        jac, _ = jax.jacfwd(self._forward_pos, has_aux=True)(angle)
        return jac[-1]

    def _forward_pos(self, angle):
        """Separated from `forward_kinematics` for use in `effector_jac`."""
        angle_sum = jnp.cumsum(angle)  # links
        length_components = self.l * jnp.array(
            [jnp.cos(angle_sum), jnp.sin(angle_sum)]
        )  # xy, links
        xy_pos = jnp.cumsum(length_components, axis=1)  # xy, links
        return xy_pos.T, length_components

    @jax.named_scope("fbx.TwoLinkArm.forward_kinematics")
    def forward_kinematics(self, state: TwoLinkArmState) -> CartesianState:
        """Return the Cartesian state of the joints and end effector, given
        the arm's configuration state.

        Arguments:
            state: The configuration state of the arm.
        """
        xy_pos, length_components = self._forward_pos(state.angle)

        ang_vel_sum = jnp.cumsum(state.d_angle)  # links
        xy_vel = jnp.cumsum(
            SINCOS_GRAD_SIGNS[1] * length_components[:, ::-1] * ang_vel_sum, axis=1
        ).T  # xy, links

        # TODO: add force conversion?
        # It might be very important NOT to convert force here.
        # Otherwise we could get positive feedback where the effector force
        # is passed to the next time step and converted to torques, then
        # converted to effector forces and added back to the torques, etc.

        return CartesianState(
            pos=xy_pos,
            vel=xy_vel,
            force=jnp.zeros_like(xy_vel),
        )

    def effector(self, state: TwoLinkArmState) -> CartesianState:
        """Return the Cartesian state of the endpoint of the arm.

        Arguments:
            state: The configuration state of the arm.
        """
        return jt.map(
            lambda x: x[-1],  # last link
            self.forward_kinematics(state),
        )

    @property
    def bounds(self) -> StateBounds[TwoLinkArmState]:
        """Suggested bounds on the arm state.

        !!! ref "Source"
            Joint angle limits adopted from [MotorNet](https://github.com/OlivierCodol/MotorNet/blob/9a56e5670d31b06fd0d81932e9bc6a8b1e46ec4b/motornet/skeleton.py#L336).
        """
        return StateBounds(
            low=TwoLinkArmState(
                angle=jnp.array([0.0, 0.0]),
                d_angle=None,
                torque=None,
            ),
            high=TwoLinkArmState(
                angle=jnp.deg2rad(jnp.array([140.0, 160.0])),
                d_angle=None,
                torque=None,
            ),
        )

    def workspace_test(
        self,
        workspace: Float[Array, "bounds=2 xy=2"],
    ) -> bool:
        """Tests whether a rectangular workspace is reachable by the arm.

        !!! Note ""
            Assumes the arm's joint angles are not bounded.

        Arguments:
            workspace: The bounds of the workspace to test.
        """
        r = sum(self.l)
        lengths = jnp.sum(corners_2d(workspace) ** 2, axis=0) ** 0.5
        if jnp.any(lengths > r):
            return False
        return True
