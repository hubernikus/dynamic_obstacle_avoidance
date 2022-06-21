"""
Multiple widget-utils to simplify the make the usage of Jupyter-Notebooks more user friendly. 
"""
# Author: LukasHuber
# Github: hubernikus
# Created:  2019-06-01
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import FloatSlider, IntSlider
import ipywidgets as widgets


from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.containers import GradientContainer

from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving

# from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
# from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import *

# from dynamic_obstacle_avoidance.visualization.animated_simulation import *
from dynamic_obstacle_avoidance.visualization.animated_simulation_ipython import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *


# Command to automatically reload libraries -- in ipython before exectureion
saveFigures = False


def widget_ellipses_vectorfield(
    x1=2,
    x2=2,
    a1=1,
    a2=1,
    p1=1,
    p2=1,
    orientation=0,
    sf=1,
    point_posX=4,
    point_posY=4,
    x_low=0.8,
    x_high=4.2,
    y_low=-2,
    y_high=2,
    draw_vectorField=True,
):

    xlim = [x_low, x_high]
    ylim = [y_low, y_high]
    pos_attractor = np.array([0, 0])

    obs = GradientContainer()
    center_position = [x1, x2]
    axes_length = [a1, a2]
    curvature = [p1, p2]
    orientation = orientation / 180 * pi
    vel = [0, 0]
    obs.append(
        Ellipse(
            axes_length=axes_length,
            curvature=curvature,
            center_position=center_position,
            orientation=orientation,
            sf=sf,
            xd=vel,
        )
    )

    point_pos = np.array([point_posX, point_posY])
    ds_init = point_pos - pos_attractor

    ds_mod = obs_avoidance_interpolation_moving(point_pos, ds_init, obs)

    fig, ax = Simulation_vectorFields(
        xlim,
        ylim,
        point_grid=10,
        obs=obs,
        pos_attractor=pos_attractor,
        figName="linearSystem_avoidanceCircle",
        noTicks=False,
        figureSize=(13.0, 10),
        draw_vectorField=draw_vectorField,
    )

    fig, ax = plt.subplots()
    # ax.quiver([point_pos[0]], [point_pos[1]], [ds_init[0]], [ds_init[1]], color='g', scale=5, zorder=10000)
    # ax.quiver([point_pos[0]], [point_pos[1]], [ds_mod[0]], [ds_mod[1]], color='r', scale=5, zorder=10)
    plt.show()
    # ax_init.quiver(point_pos[0], point_pos[1], ds_init[0], ds_init[1], c='b')


def widgetFunction_referencePoint(
    x1=2,
    x2=2,
    orientation=0,
    refPoint_dir=0,
    refPoint_rat=0,
    x_low=0.8,
    x_high=4.2,
    y_low=-2,
    y_high=2,
    draw_style="None",
):

    x_lim = [x_low, x_high]
    y_lim = [y_low, y_high]

    pos_attractor = np.array([-12, 7.5])

    sf = 1.3
    a1, a2 = 6, 1.2
    p1, p2 = 1, 1

    obs = GradientContainer()
    center_position = [x1, x2]
    axes_length = [a1, a2]
    curvature = [p1, p2]
    orientation = orientation / 180 * pi
    vel = [0, 0]

    obs.append(
        Ellipse(
            axes_length=axes_length,
            curvature=curvature,
            center_position=center_position,
            orientation=orientation,
            sf=sf,
            xd=vel,
        )
    )

    refPoint_dir *= pi / 180.0
    obs[0].local_reference_point = np.array(
        [
            obs[0].axes_length[0] * np.sqrt(refPoint_rat) * np.cos(refPoint_dir),
            obs[0].axes_length[1] * np.sqrt(refPoint_rat) * np.sin(refPoint_dir),
        ]
    )

    # rotationMatrix = np.array([[np.cos(orientation), np.sin(orientation)],
    # [-np.sin(orientation), np.cos(orientation)]])

    # obs[0].center_dyn = (rotationMatrix.T
    # @ obs[0].center_dyn*obs[0].sf + obs[0].center_position)

    if draw_style == "Simulated streamline":
        n_points = 6
        points_init = np.vstack(
            (np.ones(n_points) * x_lim[1], np.linspace(y_lim[0], y_lim[1], n_points))
        )

        points_init = points_init[:, 1:-1]

        print(obs[0].global_reference_point)
        Simulation_vectorFields(
            x_lim,
            y_lim,
            point_grid=70,
            obs=obs,
            pos_attractor=pos_attractor,
            figName="linearSystem_avoidanceCircle",
            noTicks=False,
            automatic_reference_point=False,
            figureSize=(13.0, 10),
            draw_vectorField=False,
            points_init=points_init,
        )

    elif draw_style == "Vectorfield":
        Simulation_vectorFields(
            x_lim,
            y_lim,
            point_grid=70,
            obs=obs,
            pos_attractor=pos_attractor,
            automatic_reference_point=False,
            figName="linearSystem_avoidanceCircle",
            noTicks=False,
            figureSize=(13.0, 10),
            draw_vectorField=True,
        )

    else:
        Simulation_vectorFields(
            x_lim,
            y_lim,
            point_grid=70,
            obs=obs,
            pos_attractor=pos_attractor,
            automatic_reference_point=False,
            figName="linearSystem_avoidanceCircle",
            noTicks=False,
            figureSize=(13.0, 10),
            draw_vectorField=False,
        )


class WidgetClass_intersection:
    def __init__(self, x_lim, y_lim, pos_attractor=[12, 0]):
        self.obs = GradientContainer()
        self.obs.append(
            Ellipse(
                axes_length=[3, 1],
                curvature=[1, 1],
                center_position=[-14, 4],
                orientation=45 / 180 * pi,
                sf=2.0,
            )
        )
        self.obs.append(
            Ellipse(
                axes_length=[3, 1],
                curvature=[1, 1],
                center_position=[-3, 10],
                orientation=0 / 180 * pi,
                sf=1.3,
            )
        )
        self.obs.append(
            Ellipse(
                axes_length=[2.4, 2.4],
                curvature=[4, 4],
                center_position=[-6, 4],
                orientation=-80 / 180 * pi,
                sf=1.1,
            )
        )
        self.obs.append(
            Ellipse(
                axes_length=[3, 2],
                curvature=[1, 2],
                center_position=[10, 14],
                orientation=-110 / 180 * pi,
                sf=1.5,
            )
        )

        self.x_lim = x_lim
        self.y_lim = y_lim
        self.pos_attractor = np.array(pos_attractor)

    def set_obstacle_number(self, n_obstacles=2):
        self.n_obstacles = n_obstacles

    def set_obstacle_values(
        self, it_obs, center_position_1, center_position_2, orientation
    ):

        it_obs -= 1
        # print('it obs', it_obs)
        center_position = np.copy([center_position_1, center_position_2]) * 1
        orientation = np.copy(orientation)

        self.obs[it_obs] = Ellipse(
            center_position=[center_position_1, center_position_2],
            orientation=orientation,
            axes_length=self.obs[it_obs].axes_length,
            curvature=self.obs[it_obs].curvature,
            sf=self.obs[it_obs].sf,
        )

        # for ii in range(len(self.obs)):
        # print('obs ', ii)
        # print('obs center_position ', self.obs[ii].center_position)
        # print('obs thr ', self.obs[ii].orientation)

    def update(self, check_vectorfield=True):
        obs_cp = self.obs[: self.n_obstacles]
        obs_cp = GradientContainer(obs_list=obs_cp)

        Simulation_vectorFields(
            self.x_lim,
            self.y_lim,
            point_grid=70,
            obs=obs_cp,
            pos_attractor=self.pos_attractor,
            figName="linearSystem_avoidanceCircle",
            noTicks=False,
            figureSize=(13.0, 10),
            draw_vectorField=check_vectorfield,
            show_obstacle_number=True,
        )


def run_obstacle_description():
    #%matplotlib gtk
    x_lim = [-16, 16]
    y_lim = [-2, 18]

    x1_widget = FloatSlider(
        description="Position \( x_1\)", min=x_lim[0], max=x_lim[1], step=0.1, value=6
    )
    x2_widget = FloatSlider(
        description="Position \( x_2\)", min=y_lim[0], max=y_lim[1], step=0.1, value=8
    )

    axis_widget1 = FloatSlider(
        description="Axis length 1", min=0.1, max=8, step=0.1, value=5
    )
    axis_widget2 = FloatSlider(
        description="Axis length 2", min=0.1, max=8, step=0.1, value=3
    )

    curvature_widget1 = IntSlider(description="Curvature 1", min=1, max=5, value=3)
    curvature_widget2 = IntSlider(description="Curvature 2", min=1, max=5, value=1)

    margin_widget = FloatSlider(
        description="Safety Margin", min=1, max=3, step=0.1, value=1.2
    )

    angle_widget = FloatSlider(
        description="Orientation", min=-180, max=180, step=1, value=30
    )

    pointX_widget = FloatSlider(
        description="Point position x", min=x_lim[0], max=x_lim[1], step=0.1, value=-3
    )
    pointY_widget = FloatSlider(
        description="Point position y", min=y_lim[0], max=y_lim[1], step=0.1, value=15
    )

    print("Change parameters and press <<Run Interact>> to apply.")

    interact_manual(
        widget_ellipses_vectorfield,
        x1=x1_widget,
        x2=x2_widget,
        orientation=angle_widget,
        a1=axis_widget1,
        a2=axis_widget2,
        p1=curvature_widget1,
        p2=curvature_widget2,
        sf=margin_widget,
        draw_vectorField=True,
        point_posX=pointX_widget,
        point_posY=pointY_widget,
        x_low=fixed(x_lim[0]),
        x_high=fixed(x_lim[1]),
        y_low=fixed(y_lim[0]),
        y_high=fixed(y_lim[1]),
    )


def example_reference_point():
    x_lim, y_lim = [-16, 16], [-2, 18]

    style = {"description_width": "initial"}
    # Interactive Widgets
    x1_widget = FloatSlider(
        description="Ellipse center \( x_1\)",
        min=x_lim[0],
        max=x_lim[1],
        step=0.1,
        value=6,
        style=style,
    )

    x2_widget = FloatSlider(
        description="Ellipse center \( x_2\)",
        min=y_lim[0],
        max=y_lim[1],
        step=0.1,
        value=5,
        style=style,
    )
    angle_widget = FloatSlider(
        description="Ellipse orientation \( \Theta \)",
        min=-180,
        max=180,
        step=1,
        value=30,
        style=style,
    )
    referencePoint_direction = FloatSlider(
        description="Reference point: Direction",
        min=-180,
        max=180,
        step=1,
        value=0,
        style=style,
    )
    referencePoint_excentricity = FloatSlider(
        description="Reference point: Excentricity",
        min=0,
        max=0.999,
        step=0.01,
        value=0.0,
        style=style,
    )

    draw_style = widgets.Dropdown(
        options=["None", "Vectorfield", "Simulated streamline"],
        value="Simulated streamline",
        description="Visualization",
        disabled=False,
    )

    # Main function
    interact_manual(
        widgetFunction_referencePoint,
        x1=x1_widget,
        x2=x2_widget,
        orientation=angle_widget,
        draw_style=draw_style,
        refPoint_dir=referencePoint_direction,
        refPoint_rat=referencePoint_excentricity,
        x_low=fixed(x_lim[0]),
        x_high=fixed(x_lim[1]),
        y_low=fixed(y_lim[0]),
        y_high=fixed(y_lim[1]),
    )

    print("")
    # Change parameters and press <<Run Interact>> to apply.


def choose_obstacles_number(n_obstacles, WidgetClass, x_lim, y_lim):
    # Main function
    print("Modify the parameters for the obstacle")
    it_obs = widgets.Dropdown(
        options=[ii + 1 for ii in range(n_obstacles)],
        value=1,
        description="Iterator:",
        disabled=False,
    )
    WidgetClass.set_obstacle_number(n_obstacles)

    style = {"description_width": "initial"}
    center1_widget1 = FloatSlider(
        description="Center Position \( x_1\)",
        min=x_lim[0],
        max=x_lim[1],
        step=0.1,
        value=-14,
        style=style,
    )
    center2_widget1 = FloatSlider(
        description="Center Position \( x_2\)",
        min=y_lim[0],
        max=y_lim[1],
        step=0.1,
        value=4,
        style=style,
    )
    angle_widget1 = FloatSlider(
        description="Orientation \( \Theta \)",
        min=-180,
        max=180,
        step=1,
        value=45,
        style=style,
    )

    interact(
        WidgetClass.set_obstacle_values,
        it_obs=it_obs,
        center_position_1=center1_widget1,
        center_position_2=center2_widget1,
        orientation=angle_widget1,
    )


def example_intersecting_obstacles():
    x_lim, y_lim = [-16, 16], [-2, 18]

    # Interactive Widgets
    n_obs_widget = widgets.Dropdown(
        options=[2, 3, 4],
        value=2,
        description="#",
        disabled=False,
    )

    print("Choose the number of obstacles:")

    WidgetClass = WidgetClass_intersection(x_lim=x_lim, y_lim=y_lim)
    interact(
        choose_obstacles_number,
        n_obstacles=n_obs_widget,
        WidgetClass=fixed(WidgetClass),
        x_lim=fixed(x_lim),
        y_lim=fixed(y_lim),
    )

    check_vectorfield = widgets.Checkbox(
        value=False, description="Draw Vectorfield", disabled=False
    )

    interact_manual(WidgetClass.update, check_vectorfield=check_vectorfield)


def example_dynamic_modulation():
    x_range, y_range = [-16, 16], [-2, 18]
    x_init = samplePointsAtBorder_ipython(
        number_of_points=10, x_range=x_range, y_range=y_range
    )

    x_init = np.zeros((2, 1))
    x_init[:, 0] = [8, 1]
    obs = GradientContainer()
    center_position = [-3, 8]
    axes_length = [2, 5]
    curvature = [1, 1]
    orientation = 0 / 180 * pi
    vel = [0, 0]
    obs.append(
        Ellipse(
            axes_length=axes_length,
            curvature=curvature,
            center_position=center_position,
            orientation=orientation,
            sf=1,
        )
    )

    center_position = [3, 4]
    axes_length = [3, 4]
    curvature = [1, 1]
    orientation = 0 / 180 * pi
    vel = [0, 0]
    obs.append(
        Ellipse(
            axes_length=axes_length,
            curvature=curvature,
            center_position=center_position,
            orientation=orientation,
            sf=1,
        )
    )

    ani = run_animation_ipython(
        x_init,
        obs=obs,
        x_range=x_range,
        y_range=y_range,
        dt=0.05,
        N_simuMax=1000,
        convergenceMargin=0.3,
        sleepPeriod=0.001,
        RK4_int=True,
        hide_ticks=False,
        return_animationObject=True,
        figSize=(9.5, 7),
        show_obstacle_number=True,
    )
    plt.ion()
    ani.show()

    style = {"description_width": "initial"}

    velx_widget = FloatSlider(
        description="Linear velocity \( \dot  x_1\)",
        min=-5.0,
        max=5.0,
        step=0.1,
        value=0,
        style=style,
    )

    vely_widget = FloatSlider(
        description="Linear velocity \( \dot  x_2\)",
        min=-5.0,
        max=5.0,
        step=0.1,
        value=0,
        style=style,
    )

    velAng_widget = FloatSlider(
        description="Angular velocity \( \omega\)",
        min=-5.0,
        max=5.0,
        step=0.1,
        value=0,
        style=style,
    )

    # obs_number = widgets.Dropdown(options=[ii+1 for ii in range(len(obs))],value=0, description='#', disabled=False)
    obs_number = widgets.Dropdown(
        options=[ii + 1 for ii in range(len(obs))],
        value=1,
        description="#",
        disabled=False,
    )

    print("The video can be paused and continued by pressing onto the image.")
    print("")
    print("Modify the parameters for the obstacle:")

    interact_manual(
        ani.set_velocity,
        obs_number=obs_number,
        vel_x=velx_widget,
        vel_y=vely_widget,
        vel_rot=velAng_widget,
        iteration_at_one=fixed(True),
    )
