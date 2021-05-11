#!/USSR/bin/python3
''' Script to show lab environment on computer '''

import warnings
import copy
import sys
import os

# from functools import cached_property

# import sympy.geometry as geom
# import shapely.geometry as sg
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry.polygon import LinearRing

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib import ticker
# , cm

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import Simulation_vectorFields  #
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import plt_speed_line_and_qolo
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import Cuboid
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import *
from dynamic_obstacle_avoidance.obstacle_avoidance.comparison_algorithms import obs_avoidance_potential_field, obs_avoidance_orthogonal_moving
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import obs_avoidance_interpolation_moving, obs_check_collision_2d
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import GradientContainer
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import linear_ds_max_vel
from dynamic_obstacle_avoidance.obstacle_avoidance.metric_evaluation import MetricEvaluator
from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import angle_is_in_between, angle_difference_directional

from dynamic_obstacle_avoidance.lru_chached_property import lru_cached_property
from dynamic_obstacle_avoidance.lru_chached_property import cached_property

rel_path = os.path.join(".", "scripts")
if not rel_path in sys.path:
    sys.path.append(rel_path)

from comparison_algorithms import ObstacleAvoidanceAgent, position_is_in_free_space

__author__ = "LukasHuber"
__date__ = "2020-01-15"
__email__ = "lukas.huber@epfl.ch"


class BoundaryCuboidWithGaps(Cuboid):
    ''' 2D boundary obstacle which allows to include doors (e.g. rooms).
    Currently only implemented for one door [can be extend in the future]. '''
    def __init__(self, *args, gap_points_absolute=None, gap_points_relative=None, **kwargs):
        kwargs['is_boundary'] = True
        kwargs['tail_effect'] = False
        # kwargs['sticky'] = True
        
        if (sys.version_info > (3, 0)):     # TODO: remove in future
            super().__init__(*args, **kwargs)
        else:
            super(BoundaryCuboidWithGaps, **kwargs)

        if gap_points_absolute is not None:
            self._gap_points = absolute_gap_points - np.tile(self.center_position, (absolute_gap_points.shape[1], 1)).T
        elif gap_points_relative is not None:
            self._gap_points = np.array(gap_points_relative)
        else:
            warnings.warn('No gap array assigned')
            
        self.guiding_reference_point = True
        
        # self.boundary_line = LineString(
            # [tuple(self.edge_points[:, ii]) for ii in range(self.edge_points.shape[1])])
                                        # self.boundary_points_local)
        self.has_gap_points = True

    @property
    def gap_center(self):
        return np.mean(self._gap_points, axis=1)
    
    @property
    def local_gap_center(self):
        return self.gap_center

    @lru_cached_property(maxsize=1, arg_name_list=['gap_points'])
    # @lru_cached_property
    def this_lru_cache(self):
        if not hasattr(self, '_this_lru_cache'):
            self._this_lru_cache = 0
        self._this_lru_cache += 1
        # return np.mean(self._gap_points, axis=1)
        return self._this_lru_cache

    @cached_property
    def this_cache(self):
        if not hasattr(self, '_this_cache'):
            self._this_cache = 0
        self._this_cache += 1
        # return np.mean(self._gap_points, axis=1)
        return self._this_cache
        
    @property
    def gap_angles(self):
        ''' Gap angles from center to hole in 2D.'''
        # TODO: DIY-cache lookup decorator
        # Check if cache already exists
        args_list = [self._gap_points]

        print('Get cache')
        print(self.this_cache)

        print('Get lru cache')
        print(self.this_lru_cache)
        
        breakpoint()

        if hasattr(self, '_gap_angles_arg_cache'):
            result_from_cache = True
            for ii in range(len(args_list)):
                if type(args_list[ii]) is np.ndarray:
                    if not np.all(args_list[ii] == self._gap_angles_arg_cache[ii]):
                        result_from_cache = False
                        break
                else:
                    if not all(args_list[ii] == self._gap_angles_arg_cache[ii]):
                        result_from_cache = False
                        break

            if result_from_cache:
                return self._gap_angles
        
        self._gap_angles_arg_cache = args_list

        gap_angles = np.zeros(2)
        for ii in range(gap_angles.shape[0]):
            gap_angles[ii] = np.arctan2(self._gap_points[1, ii], self._gap_points[0, ii])

        if angle_difference_directional(gap_angles[1], gap_angles[0]):
            warnings.warn('Angles wrong why around.')
            gap_angles[1], gap_angles[0] = gap_angles[0], gap_angles[1]

        self._gap_angles = gap_angles
        # TODO: test(!)
        return self._gap_angles
    
    @property
    def boundary_line(self):
        # DYI-cache lookup
        args_list = [self.edge_points]
        
        if hasattr(self, '_boundary_line_arg_cache'):
            result_from_cache = True
            for ii in range(len(args_list)):
                if type(args_list[ii]) is np.ndarray:
                    if not np.all(args_list[ii] == self._boundary_line_arg_cache[ii]):
                        result_from_cache = False
                        break
                else:
                    if not all(args_list[ii] == self._boundary_line_arg_cache[ii]):
                        result_from_cache = False
                        break

            if result_from_cache:
                return self._boundary_line
            
        self._boundary_line_arg_cache = copy.deepcopy(args_list)

        self._boundary_line =  LinearRing(
            [tuple(self.edge_points[:, ii]) for ii in range(self.edge_points.shape[1])]
        )
        return self._boundary_line
    
    def get_global_gap_points(self):
        return self.transform_relative2global(self._gap_points)
    
    def get_global_gap_center(self):
        return self.transform_relative2global(self.gap_center)

    def get_deformation_velocity(self, position, in_global_frame=False):
        ''' Get deformatkion velocity. '''
        if in_global_frame:
            position = self.transform_global2relative(position)

        # At center / reference point (no velocity.
        # Smootheness follows from Gamma -> infinity at the same time.
        if not np.linalg.norm(position):   
            return np.zeros(self.dim)
            
        normal_direction = self.get_normal_direction(
            position, in_global_frame=False, normalize=True)
        deformation_velocity = (-1)*normal_direction * self.expansion_speed_axes
        
        if in_global_frame:
            deformation_velocity = self.transform_global2relative(deformation_velocity)

        return deformation_velocity
        
    def update_step(self, delta_time):
        ''' Update position & orientation.'''
        self.update_deforming_obstacle(delta_time)
        
        if self.linear_velocity is not None:
            self.center_position = self.center_position + self.linear_velocity*delta_time

        if self.angular_velocity is not None:
            self.orientation = self.center_position + self.linear_velocity*delta_time

    def update_deforming_obstacle(self, delta_time):
        ''' Update if obstacle is deforming.'''
        # print('gap angles before', self.gap_angles)
        # import pdb; pdb.set_trace()
        self._gap_points = self._gap_points + self.expansion_speed_axes*0.5
        # print('gap angles afer', self.gap_angles)
        
        # import pdb; pdb.set_trace()
        # self.update_gap_angles()
        
        if (sys.version_info > (3, 0)):
            super().update_deforming_obstacle(delta_time)
        else:
            super(BoundaryCuboidWithGaps, self).update_deforming_obstacle(delta_time)

    def get_gamma(self, position, in_global_frame=False, gamma_distance=None):
        ''' Caclulate Gamma for 2D-Wall Case with selected Reference-Point.'''
        if in_global_frame:
            position = self.transform_global2relative(position)
            
        ref_point = self.get_projected_reference(position, in_global_frame=False)

        ref_dir = (position - ref_point)
        ref_norm = np.linalg.norm(ref_dir)

        if not ref_norm:
            # Aligned at center. Gamma >> 1 (almost infinity)
            # return sys.float_info.max
            return 1e30
        
        max_dist = self.get_maximal_distance()

        # Create line which for sure crossed the border
        ref_line = LineString([tuple(ref_point), tuple(ref_point + ref_dir*max_dist/ref_norm)])
        
        intersec = ref_line.intersection(self.boundary_line)
        point_projected_on_surface = np.array([intersec.x, intersec.y])

        dist_surface = np.linalg.norm(point_projected_on_surface-ref_point)

        # if self.is_boundary:
        if gamma_distance is None:
            gamma = dist_surface/ref_norm
        else:
            gamma = (ref_norm-dist_surface)/dist_surface + 1
            gamma = 1.0/gamma

        # import pdb; pdb.set_trace()
        return gamma
        
    def get_reference_direction(self, position, in_global_frame=False, normalize=True):
        ''' Reference direction based on guiding reference point'''

        ref_point = self.get_projected_reference(position, in_global_frame)
        reference_direction = - (position - ref_point)

        if normalize:
            ref_norm = np.linalg.norm(reference_direction)
            if ref_norm:    # nonzero
                reference_direction = reference_direction/ref_norm

        return reference_direction
        
    def get_projected_reference(self, position, in_global_frame=True):
        position_abs = copy.deepcopy(position)
        if in_global_frame:
            position = self.transform_global2relative(position)

        # Check if in between gap-center-triangle
        position_angle = np.arctan2(position[1], position[0])

        if angle_is_in_between(position_angle, self.gap_angles[0], self.gap_angles[1]):
            # Point is in gap-center-triangle
            return position_abs
            
        elif np.linalg.norm(position-self.gap_center) >= np.linalg.norm(self.gap_center):
            # Point is further away from gap than center, -> place at center
            reference_point = np.zeros(self.dim)
        else:
            # Project on gap-center-triangle-border
            it_gap = 1 if position_angle > 0 else 0
            
            # Shapely for Center etc.
            pp = Point(self.gap_center[0], self.gap_center[1])
            cc = pp.buffer(np.linalg.norm(position-self.gap_center)).boundary
            ll = LineString([(0, 0), tuple(self._gap_points[:, it_gap])])
            intersec = cc.intersection(ll)

            try:
                reference_point = np.array([intersec.x, intersec.y])
            except AttributeError:
                # Several points / Line Object -> take closest one
                point = np.array([intersec[0].x, intersec[0].y])
                dist_closest = np.linalg.norm(point)
                ind_closest = 0
                
                for pp in range(1, len(intersec)):
                    point = np.array([intersec[pp].x, intersec[pp].y])
                    dist_new = np.linalg.norm(point)

                    if dist_new < dist_closest:
                        ind_closest = pp
                        dist_closest = dist_new
                reference_point = np.array([intersec[ind_closest].x,
                                            intersec[ind_closest].y])


        if in_global_frame:
            reference_point = self.transform_relative2global(reference_point)
            
        return reference_point


    
def put_patch_behind_gap(ax, obs_list, x_min=-10, y_margin=0):
    ''' Add a patch for a door fully on the left end of a graph.'''
    # Take the wall obstacle
    obs = obs_list[-1]

    gap_points = obs.get_global_gap_points()

    edge_points = np.zeros((2, 4))
    if y_margin:    # nonzero
        # Brown patch [cover vectorfield]
        edge_points[:, 0] = [x_min, gap_points[1, 1] + y_margin] 
        edge_points[:, 1] = gap_points[:, 1]
        edge_points[:, 2] = gap_points[:, 0]
        edge_points[:, 3] = [x_min, gap_points[1, 0] - y_margin]

        door_wall_path = plt.Polygon(edge_points.T, alpha=1.0, zorder=3)
        door_wall_path.set_color(np.array([176, 124, 124])/255.)
        ax.add_patch(door_wall_path)

    # White patch
    edge_points[:, 0] = [x_min, gap_points[1, 1]] 
    edge_points[:, 1] = gap_points[:, 1]
    edge_points[:, 2] = gap_points[:, 0]
    edge_points[:, 3] = [x_min, gap_points[1, 0]]
    
    door_wall_path = plt.Polygon(edge_points.T, alpha=1.0, zorder=3)
    door_wall_path.set_color([1, 1, 1])
    ax.add_patch(door_wall_path)
    
    
def gamma_field_room_with_door(dim=2, num_resolution=20, x_min=None, fig=None, ax=None):
    x_range = [-1, 11]
    y_range = [-6, 6]
    
    obs_list = GradientContainer()
    obs_list.append(
        BoundaryCuboidWithGaps(
            name='RoomWithDoor',
            axes_length=[10, 10],
            center_position=[5, 0],
            gap_points_relative=np.array([[-5, -1], [-5, 1]]).T
        ))
    
    # Let's move to the door
    attractor_position = obs_list['RoomWithDoor'].get_global_gap_center() 
    
    if True:
        if fig is None or ax is None:
            fig_num = 1001
            fig, ax = plt.subplots(num=fig_num, figsize=(8, 6))
        
        Simulation_vectorFields(
            x_range, y_range,  obs=obs_list,
            xAttractor=attractor_position,
            saveFigure=False,
            # obs_avoidance_func=obs_avoidance_interpolation_moving,
            # automatic_reference_point=True,
            noTicks=True,
            showLabel=False,
            show_streamplot=False, draw_vectorField=False,
            fig_and_ax_handle=(fig, ax),
            normalize_vectors=False,
        )

        x_vals = np.linspace(x_range[0], x_range[1], num_resolution)
        y_vals = np.linspace(y_range[0], y_range[1], num_resolution)

        pos = np.zeros((dim, num_resolution, num_resolution))
        gamma = np.zeros((num_resolution, num_resolution))
            
        for ix in range(num_resolution):
            for iy in range(num_resolution):
                pos[:, ix, iy] = [x_vals[ix], y_vals[iy]]

                if x_min is not None and pos[0, ix, iy] < x_min:
                    continue      #
                
                gamma[ix, iy] = obs_list['RoomWithDoor'].get_gamma(
                    pos[:, ix, iy], in_global_frame=True)

        cs = plt.contourf(pos[0, :, :], pos[1, :, :],  gamma, 
                          # np.arange(0, 3.5, 0.05),
                          # np.arange(1.0, 3.5, 0.25),
                          # vmin=1, vmax=10,
                          10**(np.linspace(0, 2, 11)),
                          extend='max', 
                          locator=ticker.LogLocator(),
                          alpha=0.6, zorder=3,
                          # cmap="YlGn_r"
                          # cmap="Purples_r"
                          cmap="gist_gray"
        )

        cbar = fig.colorbar(cs, ticks=[1, 10, 100])
        cbar.ax.set_yticklabels(['1','10','100'])
        
        # print('gamma', np.round(gamma, 1))

        gap_points = obs_list[-1].get_global_gap_points()
        # ax.plot(gap_points[0, :], gap_points[1, :], color='white', linewidth='60', zorder=2)
        ax.plot(gap_points[0, :], gap_points[1, :], color='white', linewidth='2', zorder=2)

    # Gamma elements
    if True:
        obs = obs_list[-1]
        gap_points = obs.get_global_gap_points()

        # Hyper-Cone
        for ii in range(gap_points.shape[1]):
            plt.plot([obs.center_position[0], gap_points[0, ii]], [obs.center_position[1], gap_points[1, ii]], ':',
                     # color='#A9A9A9',
                     # color='#0b1873',
                     color='#00b3fa',
                     zorder=4)

        ax.plot(obs.center_position[0], obs.center_position[1],
                'k+', linewidth=18, markeredgewidth=4, markersize=13, zorder=4)

        # Hyper(half)sphere
        n_points = 30
        angles = np.linspace(-pi/2, pi/2, n_points)
        # import pdb; pdb.set_trace()
        # circle_points = np.vstack( np.cos(angles), np.sin(angles))
        rad_gap = np.linalg.norm(obs.local_gap_center)
        ax.plot(rad_gap*np.cos(angles), rad_gap*np.sin(angles), ':',
                color='#22db12',
                zorder=4)

    return obs_list
    
def vectorfield_room_with_door(dim=2, num_resolution=20, visualize_scene=True, obs_list=None, fig=None, ax=None):
    x_range = [-1, 11]
    y_range = [-6, 6]

    if obs_list is None:
        obs_list = GradientContainer()
        obs_list.append(
            BoundaryCuboidWithGaps(
                name='RoomWithDoor',
                axes_length=[10, 10],
                center_position=[5, 0],
                gap_points_relative=np.array([[-5, -1], [-5, 1]]).T
            ))

    # Let's move to the door
    attractor_position = obs_list['RoomWithDoor'].get_global_gap_center()

    if False:
        pos = np.array([4.6, -1.5])
        xd_init = linear_ds_max_vel(pos, attractor=attractor_position)
        xd = obs_avoidance_interpolation_moving(pos, xd_init, obs_list)
        
        print('xd', xd)
        import pdb; pdb.set_trace()

    if True:
        if fig is None or ax is None:
            fig_num = 1001
            fig, ax = plt.subplots(num=fig_num, figsize=(8, 6))
        
        Simulation_vectorFields(
            x_range, y_range,  obs=obs_list,
            xAttractor=attractor_position,
            saveFigure=False,
            # obs_avoidance_func=obs_avoidance_interpolation_moving,
            noTicks=True, showLabel=False,
            # automatic_reference_point=True,
            point_grid=num_resolution,
            # show_streamplot=False,
            show_streamplot=True,
            draw_vectorField=True,
            fig_and_ax_handle=(fig, ax),
            normalize_vectors=False,
        )

        # Draw gap
        plt.plot(obs_list[-1].center_position[0], obs_list[-1].center_position[1], 'k+', markersize=10, linewidth=10)
        points = obs_list['RoomWithDoor'].get_global_gap_points()
        # ax.plot(points[0, :], points[1, :], color='#46979e', linewidth='10')
        ax.plot(points[0, :], points[1, :], color='white', linewidth='2')


def test_projected_reference(max_it=1000, delta_time=0.01, max_num_obstacles=5, dim=2, visualize_scene=True, random_seed=None):
    x_range = [-1, 11]
    y_range = [-6, 6]
    
    obs_list = GradientContainer()
    obs_list.append(
        BoundaryCuboidWithGaps(
            name='RoomWithDoor',
            axes_length=[10, 10],
            center_position=[5, 0],
            gap_points_relative=np.array([[-5, -1], [-5, 1]]).T
        ))
    
    # Let's move to the door
    attractor_position = obs_list['RoomWithDoor'].get_global_gap_center() 

        
    if True:
        fig_num = 1001
        fig, ax = plt.subplots(num=fig_num, figsize=(8, 6))
        
        Simulation_vectorFields(
            x_range, y_range,  obs=obs_list,
            xAttractor=attractor_position,
            saveFigure=False,
            # obs_avoidance_func=obs_avoidance_interpolation_moving,
            noTicks=False,
            # automatic_reference_point=True,
            # show_streamplot=False,
            draw_vectorField=False, show_streamplot=False,
            fig_and_ax_handle=(fig, ax),
            normalize_vectors=False,
        )

        # Draw gap
        plt.plot(obs_list[-1].center_position[0], obs_list[-1].center_position[1], 'k+', markersize=10, linewidth=10)
        points = obs_list['RoomWithDoor'].get_global_gap_points()
        ax.plot(points[0, :], points[1, :], color='#46979e', linewidth='10')
        ax.plot(points[0, :], points[1, :], color='white', linewidth='10')

        for jj in range(30):
            it_max = 100
            for ii in range(it_max):
                position = np.random.uniform(size=(2))
                position[0] = position[0]*(x_range[1]-x_range[0]) + x_range[0]
                position[1] = position[1]*(y_range[1]-y_range[0]) + y_range[0]

                if position_is_in_free_space(position, obs_list):
                    break

            # position = np.array([6.63636, 0.285714])
            # position = np.array([2.63636, 0.285714])

            ref_proj = obs_list['RoomWithDoor'].get_projected_reference(position)
            plt.plot(position[0], position[1], 'r+')
            plt.plot(ref_proj[0], ref_proj[1], 'g+')
            plt.plot([position[0], ref_proj[0]], [position[1], ref_proj[1]], 'k--')

            
def animation_wall_with_door(max_it=1000, delta_time=0.01, max_num_obstacles=5, dim=2, visualize_scene=True, random_seed=None):
    x_range = [-1, 11]
    y_range = [-6, 6]

    attractor_position = np.array([10, 0])

    obs_list = GradientContainer()
    obs_list.append(
        BoundaryCuboidWithGaps(
            name='RoomWithDoor',
            axes_length=[10, 10],
            center_position=[5, 0],
            gap_points_relative=np.array([[-5, -1], [-5, 1]]).T,
            angular_velocity=0.1*pi,
            expansion_speed_axes=np.array([-0.1, -0.1]),
        )
    )
    
    # Let's move to the door
    attractor_position = obs_list['RoomWithDoor'].get_global_gap_center() 
    start_position = np.array([9, 1])
    
    agents = []
    agents.append(
        ObstacleAvoidanceAgent(start_position=start_position,
                               name='Dynamic',
                               avoidance_function=obs_avoidance_interpolation_moving,
                               attractor=attractor_position,
    ))

    if visualize_scene:
        fig_num = 1001
        fig, ax = plt.subplots(num=fig_num, figsize=(8, 6))

    for ii in range(max_it):
        for obs in obs_list:
            if obs.is_deforming:
                obs.update_deforming_obstacle(delta_time=delta_time)

        ax.cla()

        attractor_position = obs_list['RoomWithDoor'].get_global_gap_center() 
        Simulation_vectorFields(
            x_range, y_range,  obs=obs_list,
            xAttractor=attractor_position,
            saveFigure=False,
            obs_avoidance_func=obs_avoidance_interpolation_moving,
            noTicks=True,
            # showLabel=True,
            # automatic_reference_point=True,
            # show_streamplot=False,
            draw_vectorField=False, show_streamplot=False,
            fig_and_ax_handle=(fig, ax),
            normalize_vectors=False,
        )
        # plt.axis('equal')
        # plt.pause(0.01)

        if False:
            plt.plot(obs_list[-1].center_position[0], obs_list[-1].center_position[1], 'k+')

        for agent in agents:
            initial_velocity = linear_ds_max_vel(
                position=agent.position, attractor=attractor_position, vel_max=1.0,
            )
            
            agent.update_step(obs_list, initial_velocity=initial_velocity)
            # agent.check_collision(obs_list)
            # import pdb; pdb.set_trace() 

            if visualize_scene:
                plt.plot(agent.position_list[0, :], agent.position_list[1, :],
                         '--', label=agent.name, zorder=5)
                # print(agent.position_list)
                
        put_patch_behind_gap(ax=ax, obs_list=obs_list, x_min=-0.97, y_margin=4)
        
        # plt.axis('equal')
        plt.pause(0.01)
        import pdb; pdb.set_trace()

        if visualize_scene and not plt.fignum_exists(fig_num):
            print(f'Simulation ended with closing of figure')
            plt.pause(0.01)
            plt.close('all')
            break
    pass


def animation_ellipse(max_it=1000, delta_time=0.01, max_num_obstacles=5, dim=2, visualize_scene=True, random_seed=None):
    x_range = [-1, 11]
    y_range = [-5, 5]

    attractor_position = np.array([10, 0])

    obs_list = GradientContainer()
    obs_list.append(
        Ellipse(
            axes_length=[2, 1],
            center_position=[5, 0],
            orientation=0,
            linear_velocity=[0, 0],
            expansion_speed_axes=[1.5, 1.5], # per axis
            tail_effect=False,
        ))

    start_position = np.array([0, 2])
    
    agents = []
    agents.append(
        ObstacleAvoidanceAgent(start_position=start_position,
                               name='Dynamic',
                               avoidance_function=obs_avoidance_interpolation_moving,
                               attractor=attractor_position,
    ))

    if visualize_scene:
        fig_num = 1001
        fig, ax = plt.subplots(num=fig_num, figsize=(8, 6))

    for ii in range(max_it):
        for obs in obs_list:
            if obs.is_deforming:
                obs.update_deforming_obstacle(delta_time=delta_time)
                
        ax.cla()

        Simulation_vectorFields(
            x_range, y_range,  obs=obs_list,
            xAttractor=attractor_position,
            saveFigure=False,
            obs_avoidance_func=obs_avoidance_interpolation_moving,
            noTicks=True,
            # automatic_reference_point=True,
            # show_streamplot=False,
            draw_vectorField=False, show_streamplot=False,
            fig_and_ax_handle=(fig, ax),
            normalize_vectors=False,
        )

        if True:
            plt.plot(obs_list[-1].center_position[0], obs_list[-1].center_position[1], 'k+')

        for agent in agents:
            initial_velocity = linear_ds_max_vel(
                position=agent.position, attractor=attractor_position, vel_max=1.0)
            agent.update_step(obs_list, initial_velocity=initial_velocity)
            # agent.check_collision(obs_list)

            if visualize_scene:
                plt.plot(agent.position_list[0, :], agent.position_list[1, :],
                             '--', label=agent.name)

        # plt.axis('equal')
        plt.pause(0.01)

        if visualize_scene and not plt.fignum_exists(fig_num):
            print(f'Simulation ended with closing of figure')
            plt.pause(0.01)
            plt.close('all')
            break


if (__name__) == "__main__":
    obs_list = None
    # animation_ellipse()
    # test_projected_reference()
    
    animation_wall_with_door()
    
    # 2 plots with vectorfield & Gamma-value of boundary-region
    if False:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        plt.subplots_adjust(wspace=-0.4)

        # points_frac = 0.1
        points_frac = 1.0
        
        # fig, ax2 = plt.subplots()
        obs_list = gamma_field_room_with_door(num_resolution=int(points_frac*80), x_min=0, fig=fig, ax=ax2)
        put_patch_behind_gap(ax=ax2, obs_list=obs_list, x_min=-0.97)
        # plt.savefig('figures/' + 'boundary_with_gap_gamma' + '.png', bbox_inches='tight')
        
        ax2.text(x=1.2, y=-0.1, s=r'$\mathcal{G}$', fontsize='xx-large', color='#00b3fa')

        # fig, ax1 = plt.subplots()
        line = plt_speed_line_and_qolo(
            points_init=np.array([9, 3]),
            attractorPos=obs_list[-1].get_global_gap_center(), obs=obs_list,
            fig_and_ax_handle=(fig, ax1), dt=0.02, line_color='#22db12')
        
        vectorfield_room_with_door(num_resolution=int(points_frac*100), fig=fig, ax=ax1, obs_list=obs_list)
        put_patch_behind_gap(ax=ax1, obs_list=obs_list, x_min=-0.97, y_margin=4)
        # plt.savefig('figures/' + 'boundary_with_gap' + '.png', bbox_inches='tight')

        plt.savefig('figures/' + 'boundary_with_gap_subplot' + '.png', bbox_inches='tight')
