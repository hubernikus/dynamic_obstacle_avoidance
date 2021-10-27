""" Container to describe obstacles & wall environemnt"""
# Author Lukas Huber
# Mail lukas.huber@epfl.ch
# Created 2021-06-22
# License: BSD (c) 2021
import warnings

import numpy as np

from vartools.dynamical_systems import (
    ConstantValue,
    LinearSystem,
    LocallyRotated,
)
from vartools.directional_space import get_angle_space
from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import BaseContainer


class RotationContainer(BaseContainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ConvergenceDynamics = [None for ii in range(len(self))]

    def append(self, value):
        super().append(value)
        self._ConvergenceDynamics.append(None)

    def __delitem__(self, key):
        """Obstacle is not part of the workspace anymore."""
        super().__delitem__(self._obstacle_list[key])
        del self._ConvergenceDynamics

    def __setitem__(self, key, value):
        super().__setitem__(self._obstacle_list[key])
        self._ConvergenceDynamics[key] = None

    def set_convergence_directions(
        self, NonlinearDynamcis=None, ConvergingDynamics=None
    ):
        """Define a convergence direction / mode.
        It is implemented as 'locally-linear' for a multi-boundary-environment.

        Parameters
        ----------
        attractor_position: if non-none value: linear-system is chosen as desired function
        dynamical_system: if non-none value: linear-system is chosen as desired function
        """
        if ConvergingDynamics is not None:
            # Converging dynamics are for all the same
            self._ConvergenceDynamics = [
                ConvergingDynamics for ii in range(self.n_obstacles)
            ]
            return

        if NonlinearDynamcis.attractor_position is None:
            # WARNING: non-knowlege about local topology leads to weird behavior (!)
            for it_obs in range(self.n_obstacles):
                position = self[it_obs].center_position
                local_velocity = NonlinearDynamcis.evaluate(position)
                if np.linalg.norm(local_velocity):  # Nonzero
                    self._ConvergenceDynamics[it_obs] = ConstantValue(
                        velocity=local_velocity
                    )
                else:
                    # Make converge towards center
                    self._ConvergenceDynamics[it_obs] = LinearSystem(
                        attractor_position=position
                    )
        else:
            attractor = NonlinearDynamcis.attractor_position

            for it_obs in range(self.n_obstacles):
                position = self[it_obs].center_position
                local_velocity = NonlinearDynamcis.evaluate(position)
                if np.linalg.norm(local_velocity):
                    # Nonzero / not at attractor
                    reference_radius = self[it_obs].get_reference_length()

                    ds_direction = get_angle_space(
                        direction=local_velocity,
                        null_direction=(attractor - position),
                    )

                    self._ConvergenceDynamics[it_obs] = LocallyRotated(
                        max_rotation=ds_direction,
                        influence_pose=ObjectPose(position=position),
                        influence_radius=reference_radius,
                        attractor_position=attractor,
                    )
                else:
                    # Make it converge to attractor either way, as evaluation might be numerically bad.
                    self._ConvergenceDynamics[it_obs] = LinearSystem(
                        attractor_position=attractor
                    )

    def get_convergence_direction(self, position, it_obs):
        """Return 'convergence direction' at input 'position'."""
        return self._ConvergenceDynamics[it_obs].evaluate(position)

    def get_intersection_position(self, it_ob):
        """Get the position where two boundary-obstacles intersect."""
        if hasattr(self._ConvergenceDynamics[it_obs], "attractor_position"):
            return self._ConvergenceDynamics[it_obs].attractor_position

        else:
            raise NotImplementedError(
                "Create 'intersection-with-surface' from local DS"
            )
