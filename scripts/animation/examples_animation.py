#!/usr/bin/python3

"""
Dynamic Simulation - Obstacle Avoidance Algorithm

@author LukasHuber
@date 2018-05-24

"""

import sys

import numpy as np
from numpy import pi

import datetime

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import (
    ObstacleContainer,
)

# from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import Polygon
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *
from dynamic_obstacle_avoidance.visualization.animated_simulation import (
    run_animation,
    samplePointsAtBorder,
)

print(" ----- Script <<dynamic simulation>> started. ----- ")
#############################################################
# Choose a simulation between 0 and 12
simulationNumber = 1

saveFigures = False
#############################################################


def main(simulationNumber=0, saveFigures=False):
    if simulationNumber == 0:
        N = 10
        x_init = np.vstack((np.ones(N) * 20, np.linspace(-10, 10, num=N)))

        ### Create obstacle
        obs = []
        a = [5, 2]
        p = [1, 1]
        x0 = [10.0, 0]
        th_r = 30 / 180 * pi
        sf = 1.0

        w = 0
        x_start = 0
        x_end = 2
        # obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd, x_start=x_start, x_end=x_end, w=w))

        a = [3, 2]
        p = [1, 1]
        x0 = [7, -6]
        th_r = -40 / 180 * pi
        sf = 1.0

        xd = [0.25, 1]
        w = 0
        x_start = 0
        x_end = 10

        obs.append(
            Ellipse(
                a=a,
                p=p,
                x0=x0,
                th_r=th_r,
                sf=sf,
                xd=xd,
                x_start=x_start,
                x_end=x_end,
                w=w,
            )
        )
        a = [3, 2]
        p = [1, 1]
        x0 = [7, -6]
        th_r = -40 / 180 * pi
        sf = 1.0

        xd = [0.0, 0]
        w = 0
        x_start = 0
        x_end = 0
        # obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=xd, x_start=x_start, x_end=x_end, w=w))

        ob2 = Ellipse(
            a=[1, 1],
            p=[1, 1],
            x0=[10, -8],
            th_r=-40 / 180 * pi,
            sf=1,
            xd=[0, 0],
            x_start=0,
            x_end=0,
            w=0,
        )
        # obs.append(ob2)

        ob3 = Ellipse(
            a=[1, 1],
            p=[1, 1],
            x0=[14, -2],
            th_r=-40 / 180 * pi,
            sf=1,
            xd=[0, 0],
            x_start=0,
            x_end=0,
            w=0,
        )
        obs.append(ob3)

        x_range = [-1, 20]
        y_range = [-10, 10]
        zRange = [-10, 10]
        # obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        attractorPos = [0, 0]

        animationName = "animation_movingCircle.mp4"
        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.05,
            N_simuMax=1040,
            convergenceMargin=0.3,
            sleepPeriod=0.01,
            attractorPos=attractorPos,
            animationName=animationName,
            saveFigure=saveFigures,
        )

    if simulationNumber == 1:
        # Parallel ellipses; flow going through
        xAttractor = np.array([0, 0])

        th_r0 = 38 / 180 * pi
        obs = []
        obs.append(Ellipse(a=[4, 0.4], p=[1, 1], x0=[0, 2], th_r=30 / 180 * pi, sf=1.0))

        n = 0
        rCent = 3
        # obs[n].center_dyn=[obs[n].x0[0],
        # obs[n].x0[1]]
        obs[n].reference_point = [
            obs[n].x0[0] - rCent * np.cos(obs[n].th_r),
            obs[n].x0[1] - rCent * np.sin(obs[n].th_r),
        ]

        # n = 1
        # obs[n].center_dyn=[obs[n].x0[0]-rCent*np.cos(obs[n].th_r),
        #                    obs[n].x0[1]-rCent*np.sin(obs[n].th_r)]

        x_range = [-5, 5]
        y_range = [-1, 7]
        N = 20

        x_init = samplePointsAtBorder(N, x_range, y_range)

        animationName = "test.mp4"
        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.02,
            N_simuMax=1000,
            convergenceMargin=0.3,
            sleepPeriod=0.001,
            RK4_int=True,
            animationName=animationName,
            saveFigure=False,
        )

    elif simulationNumber == 2:
        x_range = [-0.7, 0.3]
        y_range = [2.3, 3.0]

        x_range = [-3, 3]
        y_range = [-3, 3.0]

        N = 10
        # x_init = np.vstack((np.linspace(-.19,-0.16,num=N),
        # np.ones(N)*2.65))

        x_init = np.vstack((np.linspace(-3, -1, num=N), np.ones(N) * 0))

        xAttractor = np.array([0, 0])

        obs = []

        obs.append(
            Ellipse(a=[1.1, 1], p=[1, 1], x0=[0.5, 1.5], th_r=-25 * pi / 180, sf=1.0)
        )

        a = [0.2, 5]
        p = [1, 1]
        x0 = [0.5, 5]
        th_r = -25 / 180 * pi
        sf = 1.0
        obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.003,
            N_simuMax=1040,
            convergenceMargin=0.3,
            sleepPeriod=0.01,
        )

    elif simulationNumber == 3:
        x_range = [-0.7, 0.3]
        y_range = [2.3, 3.0]

        x_range = [-4, 4]
        y_range = [-0.1, 6.0]

        N = 20
        x_init = np.vstack((np.linspace(-4.5, 4.5, num=N), np.ones(N) * 5.5))

        xAttractor = np.array([0, 0])

        obs = []
        obs.append(
            Ellipse(a=[1.1, 1.2], p=[1, 1], x0=[-1, 1.5], th_r=-25 / 180 * pi, sf=1.0)
        )

        obs.append(
            Ellipse(
                a=[1.8, 0.4],
                p=[1, 1],
                x0=[0, 4],
                th_r=20 / 180 * pi,
                sf=1.0,
            )
        )

        obs.append(
            Ellipse(a=[1.2, 0.4], p=[1, 1], x0=[3, 3], th_r=-30 / 180 * pi, sf=1.0)
        )

        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.02,
            N_simuMax=1040,
            convergenceMargin=0.3,
            sleepPeriod=0.01,
            RK4_int=True,
        )

    elif simulationNumber == 4:
        # Moving in LAB
        x_range = [0, 16]
        y_range = [0, 9]

        t_fact = 0.17  # Speed up / slow down simulation

        # x_init = np.vstack((np.ones(N)*16,
        #                    np.linspace(0,9,num=N) ))b

        ### Create obstacle
        obs = []
        x0 = [3.5, 1]
        p = [1, 1]
        th_r = -10
        a = [2.5 * 1.3, 0.8 * 1.3]
        sf = 1

        xd0 = [0, 0]
        w0 = 0

        x01 = x0
        x_start = 0
        x_end = 10
        obs.append(
            Ellipse(
                a=a,
                p=p,
                x0=x01,
                th_r=th_r,
                sf=sf,
                x_start=x_start,
                x_end=x_end,
                timeVariant=True,
            )
        )

        def func_w1(t):
            t_interval1 = [0, 2.5, 5, 7, 8, 10]
            w1 = [th_r, -20, -140, -140, -170, -170]

            for ii in range(len(t_interval1) - 1):
                if t < t_interval1[ii + 1] * 1 * t_fact:
                    return (
                        (w1[ii + 1] - w1[ii])
                        / (t_interval1[ii + 1] - t_interval1[ii])
                        * pi
                        / 180
                        * 1
                        / t_fact
                    )
            return 0

        def func_xd1(t):
            t_interval1x = [0, 2.5, 5, 7, 8, 10]
            xd1 = [[x01[0], 7, 9, 9, 7, 6], [x01[1], 4, 5, 5, 4, -2]]

            for ii in range(len(t_interval1x) - 1):
                if t < t_interval1x[ii + 1] * t_fact:
                    dt = (t_interval1x[ii + 1] - t_interval1x[ii]) * t_fact
                    return [
                        (xd1[0][ii + 1] - xd1[0][ii]) / dt,
                        (xd1[1][ii + 1] - xd1[1][ii]) / dt,
                    ]
            return [0, 0]

        obs[0].func_w = func_w1
        obs[0].func_xd = func_xd1

        x0 = [12, 8]
        p = [1, 1]
        th_r = 0
        sf = 1
        a = [2 * 1.3, 1.2 * 1.3]

        xd0 = [0, 0]
        w0 = 0

        x_start = 0
        x_end = 10
        obs.append(
            Ellipse(
                a=a,
                p=p,
                x0=x0,
                th_r=th_r,
                sf=sf,
                x_start=x_start,
                x_end=x_end,
                timeVariant=True,
            )
        )

        def func_w2(t):
            t_interval = [0, 2.0, 6.5, 7, 10]
            w = [th_r, -60, -60, 30, 30]

            for ii in range(len(t_interval) - 1):
                if t < t_interval[ii + 1] * t_fact:
                    return (
                        (w[ii + 1] - w[ii])
                        / (t_interval[ii + 1] - t_interval[ii])
                        * pi
                        / 180
                        * 1
                        / t_fact
                    )
            return 0

        def func_xd2(t):
            t_interval = [0, 2.0, 5, 6.5, 9, 10]
            xd = [[x0[0], 13, 13, 12, 14, 15], [x0[1], 6, 6, 3, -2, -3]]

            for ii in range(len(t_interval) - 1):
                if t < t_interval[ii + 1] * t_fact:
                    dt = (t_interval[ii + 1] - t_interval[ii]) * t_fact
                    return [
                        (xd[0][ii + 1] - xd[0][ii]) / dt,
                        (xd[1][ii + 1] - xd[1][ii]) / dt,
                    ]
            return [0, 0]

        obs[1].func_w = func_w2
        obs[1].func_xd = func_xd2

        N = 20
        x_init = samplePointsAtBorder(N, x_range, y_range)
        collisions = obs_check_collision(x_init, obs)
        x_init = x_init[:, collisions[0]]

        attractorPos = [4, 8]

        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.002,
            N_simuMax=750,
            convergenceMargin=0.3,
            sleepPeriod=0.01,
            attractorPos=attractorPos,
            hide_ticks=False,
            saveFigure=saveFigures,
            animationName="replication_humans",
        )

    elif simulationNumber == 5:
        x_range = [-4, 4]
        y_range = [-0.1, 6.0]

        N = 10

        x_init = samplePointsAtBorder(N, x_range, y_range)
        print("axample at rorder")

        xAttractor = np.array([0, 0])

        obs = []
        obs.append(
            Ellipse(a=[1.1, 1.2], p=[1, 1], x0=[-1, 1.5], th_r=-25 / 180 * pi, sf=1)
        )

        obs.append(
            Ellipse(
                a=[1.8, 0.4],
                p=[1, 1],
                x0=[0, 4],
                th_r=20 / 180 * pi,
                sf=1.0,
            )
        )

        obs.append(
            Ellipse(a=[1.2, 0.4], p=[1, 1], x0=[3, 3], th_r=-30 / 180 * pi, sf=1.0)
        )

        N = 10

        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.001,
            N_simuMax=1040,
            convergenceMargin=0.3,
            sleepPeriod=0.01,
        )

        if True:  # save animation
            anim.ani.save("ani/animation_peopleWalking.mp4", dpi=100, fps=25)
            print("Saving finished.")

        # dist slow = 0.18
        # anim.ani.save('ani/simue.mpeg', writer="ffmpeg")
        # FFwriter = animation.FFMpegWriter()
        # anim.ani.save('ani/basic_animation.mp4', writer = FFwriter, fps=20)

    if simulationNumber == 6:
        x_range = [-0.1, 12]
        y_range = [-5, 5]

        N = 5

        x_init = np.vstack(
            (
                np.ones((1, N)) * 8,
                np.linspace(-1, 1, num=N),
            )
        )

        xAttractor = [0, 0]

        obs = []
        a = [0.3, 2.5]
        p = [1, 1]
        x0 = [2, 0]
        th_r = -50 / 180 * pi
        sf = 1
        obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

        # Ellipse 2
        a = [0.4, 2.5]
        p = [1, 1]
        # x0 = [7,2]
        x0 = [6, 0]
        th_r = 50 / 180 * pi
        sf = 1
        obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))
        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.02,
            N_simuMax=1000,
            convergenceMargin=0.3,
            sleepPeriod=0.001,
            RK4_int=True,
        )

    if simulationNumber == 7:
        xAttractor = np.array([0, 0])
        centr = [2, 2.5]

        obs = []
        N = 12
        R = 5
        th_r0 = 38 / 180 * pi
        rCent = 2.4
        for n in range(N):
            obs.append(
                Ellipse(
                    a=[0.4, 3],
                    p=[1, 1],
                    x0=[R * cos(2 * pi / N * n), R * sin(2 * pi / N * n)],
                    th_r=th_r0 + 2 * pi / N * n,
                    sf=1.0,
                )
            )

            obs[n].center_dyn = [
                obs[n].x0[0] - rCent * sin(obs[n].th_r),
                obs[n].x0[1] + rCent * cos(obs[n].th_r),
            ]

            obs[n].tail_effect = True

        x_range = [-10, 10]
        y_range = [-8, 8]
        N = 20

        x_init = samplePointsAtBorder(N, x_range, y_range)

        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.01,
            N_simuMax=400,
            convergenceMargin=0.3,
            sleepPeriod=0.001,
            RK4_int=True,
            saveFigure=saveFigures,
        )

    if simulationNumber == 8:
        xAttractor = np.array([0, 0])
        centr = [2, 2.5]

        obs = []
        obs.append(
            Ellipse(
                a=[2, 2],
                p=[1, 1],
                x0=[10, -7],
                th_r=0,
                sf=1.0,
                xd=[-6, 7],
                x_start=0,
                x_end=10,
            )
        )

        ob = Ellipse(a=[0.3, 0.4], p=[1, 1], x0=[9, 4], th_r=0, sf=1.0)
        obs.append(ob)

        x_range = [-1, 10]
        y_range = [-5, 5]
        N = 20

        x_init = samplePointsAtBorder(N, x_range, y_range)

        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.01,
            N_simuMax=800,
            convergenceMargin=0.3,
            sleepPeriod=0.01,
            RK4_int=True,
            animationName="animation_ring_convergence",
        )

    if simulationNumber == 9:
        xAttractor = np.array([0, 0])
        centr = [2, 2.5]

        obs = []
        obs.append(
            Ellipse(
                a=[0.4, 3],
                p=[1, 1],
                x0=[2, 0],
                th_r=0,
                sf=1.0,
                w=3,
                x_start=0,
                x_end=10,
            )
        )

        x_range = [-3, 7]
        y_range = [-5, 5]
        N = 20

        x_init = samplePointsAtBorder(N, x_range, y_range)

        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.005,
            N_simuMax=620,
            convergenceMargin=0.3,
            sleepPeriod=0.01,
            saveFigure=saveFigures,
            animationName="rotatingEllipse",
        )

    if simulationNumber == 10:
        xAttractor = np.array([0, 0])
        centr = [2, 2.5]

        obs = []
        obs.append(
            Ellipse(
                a=[0.4, 3],
                p=[1, 1],
                x0=[2, 0],
                th_r=0,
                sf=1.0,
                w=3,
                x_start=0,
                x_end=10,
            )
        )

        obs.append(Ellipse(a=[0.4, 0.3], p=[1, 1], x0=[4, 3], th_r=0, sf=1.0))

        x_range = [-3, 7]
        y_range = [-5, 5]

        N = 20

        x_init = samplePointsAtBorder(N, x_range, y_range)

        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.005,
            N_simuMax=620,
            convergenceMargin=0.3,
            sleepPeriod=0.01,
            saveFigure=saveFigures,
            animationName="rotatingEllipseWithStaticp",
        )

    if simulationNumber == 11:
        dy = 2.5

        xAttractor = np.array([0, 0])
        centr = [2.05, 2.55 - dy]

        obs = []
        a = [0.6, 0.6]
        p = [1, 1]
        x0 = [2.0, 3.2 - dy]
        th_r = -60 / 180 * pi
        sf = 1.2
        obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))
        obs[0].center_dyn = centr

        a = [1, 0.4]
        p = [1, 3]
        x0 = [1.5, 1.6 - dy]
        th_r = +60 / 180 * pi
        sf = 1.2
        obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))
        obs[1].center_dyn = centr

        a = [1.2, 0.2]
        p = [2, 2]
        x0 = [3.3, 2.1 - dy]
        th_r = -20 / 180 * pi
        sf = 1.2
        obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))
        obs[1].center_dyn = centr

        N = 20

        x_range = [-0.5, 5.5]
        y_range = [-2.5, 2.5]

        x_init = samplePointsAtBorder(N, x_range, y_range)

        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.005,
            N_simuMax=600,
            convergenceMargin=0.3,
            sleepPeriod=0.01,
        )

    elif simulationNumber == 13:
        N = 15
        x_init = np.vstack(
            (np.ones(N) * 1, np.linspace(-1, 1, num=N), np.linspace(-1, 1, num=N))
        )
        ### Create obstacle
        obs = []

        x0 = [0.5, 0.2, 0.0]
        a = [0.4, 0.1, 0.1]
        # a = [4,4,4]
        p = [10, 1, 1]
        th_r = [0, 0, 30.0 / 180 * pi]
        sf = 1.0

        xd = [0, 0, 0]
        w = [0, 0, 0]

        x_start = 0
        x_end = 2
        obs.append(
            Ellipse(
                a=a,
                p=p,
                x0=x0,
                th_r=th_r,
                sf=sf,
                xd=xd,
                x_start=x_start,
                x_end=x_end,
                w=w,
            )
        )
        import pdb

        pdb.set_trace()  ## DEBUG ##

        ### Create obstacle
        x0 = [0.5, -0.2, 0]
        a = [0.4, 0.1, 0.1]
        p = [10, 1, 1]
        th_r = [0, 0, -30 / 180 * pi]
        sf = 1

        xd = [0, 0, 0]
        w = [0, 0, 0]

        x_start = 0
        x_end = 2
        obs.append(
            Ellipse(
                a=a,
                p=p,
                x0=x0,
                th_r=th_r,
                sf=sf,
                xd=xd,
                x_start=x_start,
                x_end=x_end,
                w=w,
            )
        )
        x_range = [-0.2, 1.8]
        y_range = [-1, 1]

        import pdb

        pdb.set_trace()  ## DEBUG ##

        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.003,
            N_simuMax=1040,
            convergenceMargin=0.3,
            sleepPeriod=0.01,
        )

    if simulationNumber == 14:
        N = 8
        x_init = np.vstack((np.ones(N) * 18, np.linspace(-7, 7, num=N)))

        ### Create obstacle
        obs = ObstacleContainer()
        obs.append(
            Ellipse(
                axes_length=[3, 2],
                p=[1, 1],
                x0=[7, -6],
                th_r=-40 / 180.0 * pi,
                xd=[0.25, 1],
                x_start=0,
                x_end=10,
                w=0,
            )
        )

        # obs.append(Ellipse(a=[1,1], p=[1,1], x0=[14,-2], th_r=-40/180*pi,
        # xd=[0, 0], x_start=0, x_end=0, w=0))

        # obs.append(Ellipse(a=[11, 12], p=[1,1], x0=[10, 0], th_r=-40/180*pi,
        # xd=[0, 0], x_start=0, x_end=0, w=0, is_boundary=True))

        edge_points = np.array([[-1, -10], [-1, 10], [20, 10], [20, -10]]).T

        obs.append(
            Polygon(
                edge_points=edge_points,
                is_boundary=True,
                xd=[0, 0],
                x_start=0,
                x_end=0,
                w=0,
            )
        )

        # obs[-1].move_center([7,0])
        # obs[-1].orientation=30/180.*pi

        x_range, y_range = [-1.5, 20.5], [-10.5, 10.5]
        zRange = [-10, 10]
        # obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

        attractorPos = [0, 0]

        animationName = "animation_boundary_square.mp4"
        run_animation(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.05,
            N_simuMax=1040,
            convergenceMargin=0.3,
            sleepPeriod=0.01,
            attractorPos=attractorPos,
            animationName=animationName,
            saveFigure=saveFigures,
        )

    print("\n\n---- Script finished at {}---- \n\n".format(datetime.datetime.now()))


if (__name__) == "__main__":

    if len(sys.argv) >= 2 and (sys.argv[1]) == "-i":
        del sys.argv[1]

    if len(sys.argv) > 1:
        simulationNumber = sys.argv[1]

    if len(sys.argv) > 2:
        saveFigures = sys.argv[2]

    try:
        main(simulationNumber=simulationNumber, saveFigures=saveFigures)
    except:
        raise
