#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="avoidance_comparison",
    version="0.1.0",
    description="Comparsion of Obstacle Avoidance Algorithms",
    author="Lukas Huber",
    author_email="lukas.huber@epfl.ch",
    # packages=['dynamic_obstacle_avoidance'],
    # packages=find_packages(include=['src', 'src/analysis/comparison']),
    package_dir={
        "": "src",
        # 'vartools': 'src/various_tools/src',
    },
    # install_requires=[
    # 'ipython',
    # 'numpy',
    # 'matplotlib',
    # 'scipy',
    # 'PyYaml',
    # 'shapely',
    # ],
    # scripts=['scripts/examples_animation.py', 'scripts/examples_vector_field.py'],
)
