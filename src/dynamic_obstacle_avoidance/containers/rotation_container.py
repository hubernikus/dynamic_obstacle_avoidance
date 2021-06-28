""" Container to describe obstacles & wall environemnt"""
# Author Lukas Huber 
# Mail lukas.huber@epfl.ch
# Created 2021-06-22
# License: BSD (c) 2021

import numpy as np
import warnings

from vartools.dynamicalsys import ConstantValue, LocallyRotated
from vartools.directional_space import get_angle_space

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
        del(self._ConvergenceDynamics)

    def __setitem__(self, key, value):
        super().__setitem__(self._obstacle_list[key])
        self._ConvergenceDynamics[key] = None
        
    def set_convergence_directions(self, DynamicalSystem):
        """ Define a convergence direction / mode.
        It is implemented as 'locally-linear' for a multi-boundary-environment.

        Parameters
        ----------
        attractor_position: if non-none value: linear-system is chosen as desired function
        dynamical_system: if non-none value: linear-system is chosen as desired function
        """
        if dynamical_system.attractor_position is None:
            for it_obs in range(self.n_obstacles):
                position = self[it_obs].center_position
                local_velocity = DynamicalSystem.evaluate(position)
                self._ConvergenceDS = ConstantValue(velocity=local_velocity)

        else:
            attractor = dynamical_system.attractor_position
            
            for it_obs in range(self.n_obstacles):
                position = self[it_obs].center_position
                local_velocity = DynamicalSystem.evaluate(position)
                reference_radius = self[it_obs].get_reference_length()
                
                ds_direction = get_angle_space(direction=local_velocity,
                                               null_direction=(attractor-position))
                
                self._ConvergenceDynamics[it_obs] = LocallyRotated(mean_rotation=ds_direction,
                                                                 rotation_center=position,
                                                                 influence_radius=reference_radius)
                

    def get_convergence_direction(self, position, it_obs):
        """ Return 'convergence direction' at input 'position'."""
        return self._ConvergenceDynamics[it_obs].evaluate(position)
