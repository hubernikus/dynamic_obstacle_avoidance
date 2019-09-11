#!/usr/bin/env python

from setuptools import setup

setup(name='dynamic_obstacle_avoidance',
      version='1.0',
      description='Dynamic Obstacle Avoidance',
      author='Lukas Huber',
      author_email='lukas.huber@epfl.ch',
      packages=['dynamic_obstacle_avoidance',
                'dynamic_obstacle_avoidance.dynamical_system',
                'dynamic_obstacle_avoidance.obstacle_avoidance',
                'dynamic_obstacle_avoidance.visualization'],
      scripts=['scripts/examples_animation.py', 'scripts/examples_vector_field.py'],
      package_dir={'': 'src'}
     )