"""
Dummy robot models for cluttered obstacle environment + testing
"""
# Author: Lukas Huber
from math import pi

from scipy.spatial.transform import Rotation

import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.obstacles import FlatPlane
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from robot_avoidance.robot_arm_avoider import RobotArmAvoider

try:
    from robot_avoidance.jacobians.model_robot_2d import _get_jacobian
except ImportError:
    print("Jacobian-file found. -- Limited functionality.")


class RobotArm:
    # Default jacobian matrix for end-effector
    def __init__(self, max_joint_velocity=pi / 2):
        self.max_joint_velocity = max_joint_velocity

    @property
    def link_lengths(self):
        return self._link_lengths

    @property
    def n_links(self):
        return self.n_joints

    def _my_jacobian(self, ll, qq):
        # Global function of get-jacobian
        return _get_jacobian(ll, qq)

    def set_joint_state(self, value, input_unit="rad"):
        if value.shape[0] != self.n_joints:
            raise Exception("Wrong dimension of joint input.")

        if input_unit == "rad":
            self._joint_state = value
        elif input_unit == "deg":
            self._joint_state = value * pi / 180.0
        else:
            raise Exception(f"Unpexpected input_unit argument: '{input_unit}'")

    def update_state(
        self,
        joint_velocity_control,
        delta_time=0.01,
        input_unit="rad",
        check_max_velocity=True,
    ):
        if input_unit == "deg":
            joint_velocity_control = joint_velocity_control * pi / 180
        elif input_unit != "rad":
            raise Exception("Unkown joint-control input.")

        if check_max_velocity:
            ind_max = np.abs(joint_velocity_control) > self.max_joint_velocity
            if any(ind_max):  # bigger than zero
                joint_velocity_control[ind_max] = np.copysign(
                    self.max_joint_velocity, joint_velocity_control[ind_max]
                )

        self._joint_state = self._joint_state + joint_velocity_control * delta_time

    def get_jacobian(self, link_lengths=None, joint_state=None):
        """Returns end-effector velocity based on current joint state."""
        # TODO: store jacobian for current state for speed
        if link_lengths is None:
            link_lengths = self._link_lengths
        if joint_state is None:
            joint_state = self._joint_state

        return self._my_jacobian(ll=link_lengths, qq=joint_state)

    def set_jacobian(self, function):
        """Reset the jacobian which takes link-lenght and joint-state as input."""
        self._my_jacobian = function


class RobotArm2D(RobotArm):
    def __init__(self, link_lengths, joint_state=None, base_position=None):
        super().__init__()
        self.name = "robot_arm_2d"
        self._link_lengths = link_lengths

        self.n_joints = self._link_lengths.shape[0]
        if joint_state is None:
            self._joint_state = np.zeros(self.n_joints)
        else:
            self._joint_state = joint_state

        self.dimension = 2

        if base_position is None:
            self.base_position = np.zeros(self.dimension)
        else:
            self.base_position = base_position

        self._joint_velocity = np.zeros(self.n_joints)

    def get_transformation_matrices(self, level=None):
        """Transformation matrices.
        Note, they are expressed in 3D to have compatibility.

        Paramteters
        -----------
        level: value to indicate 'which matrix is taken'"""
        dim = 3
        if level is None:
            level = self.n_links

        transformation_matrices = np.zeros((dim + 1, dim + 1, level + 1))
        transformation_matrices[-1, -1, :] = 1

        if level:
            # Define Last Matrix if greater than 0
            ii = level
            # transformation_matrices[:dim, -1, -1] = [self._link_lengths[-1], 0, 0]
            transformation_matrices[:dim, -1, -1] = [
                self._link_lengths[ii - 1],
                0,
                0,
            ]

            if level == self.n_links:
                transformation_matrices[:dim, :dim, -1] = np.eye(dim)
            else:
                rot_matr = Rotation.from_euler(
                    "xyz", [0, 0, self._joint_state[ii]]
                ).as_matrix()
                transformation_matrices[:dim, :dim, ii] = rot_matr

        for ii in range(1, level):
            rot_matr = Rotation.from_euler(
                "xyz", [0, 0, self._joint_state[ii]]
            ).as_matrix()
            transformation_matrices[:dim, :dim, ii] = rot_matr
            transformation_matrices[:dim, -1, ii] = [
                self._link_lengths[ii - 1],
                0,
                0,
            ]

        # First Matrix
        rot_matr = Rotation.from_euler("xyz", [0, 0, self._joint_state[0]]).as_matrix()
        transformation_matrices[:dim, :dim, 0] = rot_matr
        return transformation_matrices

    def get_ee_in_base(self):
        """Returns position of end-effector in base-frame."""
        return self.get_joint_in_base()

    def get_joint_in_base(self, level: int = None, relative_point_position: float = 0):
        """Get position of a point on the joint at 'level' with a displacement
        in x direction of 'relative_point_position' in base-frame."""
        dim = 3
        if level is None:
            level = self.n_links

        transformation_matrices = self.get_transformation_matrices(level=level)

        position = np.zeros((dim + 1))
        position[0] = relative_point_position
        position[-1] = 1

        for ii in reversed(range(level + 1)):
            position = transformation_matrices[:, :, ii] @ position

        return position[:2]

    def get_joint_orientation_in_base(self, level: int = None):
        orientation = np.sum(self._joint_state[: level + 1])
        return orientation

    def get_inverse_kinematics_at_level(
        self, linear_velocity: np.ndarray, angular_velocity: float, level: int
    ) -> np.ndarray:
        """Return array of desired joint_velocities of length 'level."""
        link_lengths = np.zeros(self.n_links)
        link_lengths[: level + 1] = self._link_lengths[: level + 1]

        joint_state = np.zeros(self.n_links)
        joint_state[: level + 1] = self._joint_state[: level + 1]

        jacobian = self.get_jacobian(
            link_lengths,
        )
        jacobian = jacobian[:, : level + 1]

        desired_joint_velocity = LA.pinv(jacobian) @ np.hstack(
            (linear_velocity, angular_velocity)
        )

        if self.max_joint_velocity is not None:
            ind_max = np.abs(desired_joint_velocity) > self.max_joint_velocity
            desired_joint_velocity[ind_max] = np.copysign(
                self.max_joint_velocity, desired_joint_velocity[ind_max]
            )

        return desired_joint_velocity

    def get_inverse_kinematics(self, desired_velocity):
        """Inverse kinematics solving."""
        jacobian = self.get_jacobian(self._link_lengths, self._joint_state)
        desired_joint_velocity = LA.pinv(jacobian[:2, :]) @ desired_velocity

        if self.max_joint_velocity is not None:
            ind_max = np.abs(desired_joint_velocity) > self.max_joint_velocity
            desired_joint_velocity[ind_max] = np.copysign(
                self.max_joint_velocity, desired_joint_velocity[ind_max]
            )

        return desired_joint_velocity

    def get_forward_kinematics(self, joint_velocity):
        """Forward kinematics solving."""
        jacobian = self.get_jacobian(self._link_lengths, self._joint_state)
        velocity = jacobian[:2, :] @ joint_velocity
        return velocity

    def get_jacobian_of_level_and_relative_position(
        self, level: int, relative_point_position: float = 0
    ) -> np.ndarray:
        if level > self.n_links:
            raise Exception(
                f"level = {level} "
                + "-> To high level for evaluation (<= {self.n_links})."
            )

        link_lengths = np.zeros(self.n_links)
        link_lengths[:level] = self._link_lengths[:level]

        if level < self.n_links:
            link_lengths[level] = relative_point_position
        elif relative_point_position:
            raise Exception("Relative position not considered for full arm-length")

        joint_state = np.zeros(self.n_links)
        joint_state[: level + 1] = self._joint_state[: level + 1]

        # breakpoint()
        return self.get_jacobian(link_lengths, joint_state)

    # def get_orientation_from_
    def get_cartesian_vel_from_joint_velocity_on_link(
        self,
        joint_velocity: np.ndarray,
        level: int,
        relative_point_position: float = 0,
    ) -> np.ndarray:
        """Returns the (cartesian 2D) velocity of a point at x-position on link at level
        and with given joint_velocity.

        Parameters
        ----------
        joint_velocity: an array with the joint velocity in rad/s
        level: an int which defines at what level of the link-chain the point is placed
        relative_point_position: the x-position at which the point is placed
            (on previously defined link); this argument only considered when
            not on the 'last' link.

        Returns
        -------
        velocity (cartesian) in 2D as array of floats
        """
        jacobian = self.get_jacobian_of_level_and_relative_position(
            level, relative_point_position
        )
        velocity = jacobian[:2, :] @ joint_velocity
        return velocity

    def get_joint_command_from_desired_cartesian_vel_on_link(
        self, velocity, level: int, relative_poin_position: float = 0
    ):
        """
        Returns control of

        see previous function for more info.
        """
        jacobian = self.get_jacobian_of_level_and_relative_position(
            level, relative_point_position
        )

        if relative_poin_position:
            level_plus_last_link = level + 1
        else:
            level_plus_last_link = level

        jacobian = jacobian[:, :level_plus_last_link]
        velocity = LA.pinv(jacobian[:2, :]) @ joint_velocity

        return velocity

    def set_velocity(self, value, input_unit="rad"):
        if input_unit == "rad":
            self._joint_velocity = value
        elif input_unit == "deg":
            self._joint_velocity = value * pi / 180.0
        else:
            raise Exception(f"Unpexpected input_unit argument: '{input_unit}'")

    def get_ee_velocity(self, joint_velocity, input_unit="rad"):
        """Return end-effectory velocity by evaluating the jacobian at current position."""
        if input_unit == "deg":
            joint_velocity = joint_velocity * pi / 180.0
        elif input_unit == "rad":
            pass
        else:
            raise Exception(f"Unpexpected input_unit argument: '{input_unit}'")

        jacobian = self.get_jacobian()
        return jacobian @ joint_velocity

    def draw_robot(
        self,
        ax,
        # link_color='orange', joint_color='black',
        link_color="#f0b01d",
        joint_color="#a37917",
        link_line_width=8,
        joint_marker_size=12,
    ):
        ax.plot(
            self.base_position[0],
            self.base_position[1],
            "o",
            markersize=joint_marker_size * 1.5,
            color=joint_color,
        )
        # joint_state_plus_0 = self.get_joint_stat_plus0()
        joint_state = self._joint_state
        pos_joint_low = self.base_position
        state_joint_low = 0

        for ii in range(self.n_links):
            # state_joint_low += self._joint_state[ii]
            state_joint_low += joint_state[ii]
            pos_joint_high = (
                pos_joint_low
                + np.array([np.cos(state_joint_low), np.sin(state_joint_low)])
                * self._link_lengths[ii]
            )

            ax.plot(
                [pos_joint_low[0], pos_joint_high[0]],
                [pos_joint_low[1], pos_joint_high[1]],
                "-",
                linewidth=link_line_width,
                color=link_color,
                zorder=1,
            )

            ax.plot(
                pos_joint_high[0],
                pos_joint_high[1],
                "o",
                # markeredgewidth=2,
                markersize=joint_marker_size,
                color=joint_color,
                zorder=2,
            )
            pos_joint_low = pos_joint_high


class ModelRobot2D(RobotArm2D):
    """
    Model Robot in 2D with Various Joints.
    """

    def __init__(self):
        self.n_joints = 4
        self._link_lengths = np.array([0.5, 1, 1, 1])
        super().__init__(link_lengths=self._link_lengths)
        # super().__init__()

        # In radiaon
        self._joint_state = np.zeros(self.n_joints)
        self._joint_axes_of_rotation = [2, 2, 2, 2]  # important for 3D
        self._joint_velocity = np.zeros(self.n_joints)

        self.base_position = np.array([0, 0])

        self.dimension = 2

        self.name = "model_robot_2d"

        # self.transformation_matrices = self.get_transformation_matrices()
        # self.total_transformation = self.get_total_transformation(self.transformation_matrices)

    # def get_joint_stat_plus0(self):
    # """ Allows for easier iteration, since 0-angle for base link is added."""
    # return np.hstack((self._joint_state, 0))
