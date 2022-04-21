from setuptools import setup

package_name = "dynamic_obstacle_avoidance"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Lukas Huber",
    maintainer_email="lukas.huber@epfl.ch",
    description="Dynamic Obstacle Avoidance",
    license="TODO",
    # package_dir={'': 'src'},
    tests_require=["pytest"],
    # entry_points={
    # 'console_scripts': ['simulation_loader = pybullet_ros2.simulation_loader:main',
    # 'pybullet_ros2 = pybullet_ros2.pybullet_ros2:main']
    # }
)
