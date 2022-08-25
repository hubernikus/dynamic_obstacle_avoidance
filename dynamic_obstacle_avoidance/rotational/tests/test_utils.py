import numpy as np

from dynamic_obstacle_avoidance.rotational.utils import project_point_onto_ray


def test_project_point_onto_ray():
    ray0 = np.array([[0, 1, 0], [1, 0, 1]]).T

    point0 = np.array([-1, 2, -1])

    point = project_point_onto_ray(point0, ray0)
    assert np.allclose(point, ray0[:, 0])

    ray1 = np.array([[1, 0, 1], [0, 1, 0]]).T
    point = project_point_onto_ray(point0, ray1)
    assert not np.allclose(point, ray1[:, 0])


if (__name__) == "__main__":
    test_project_point_onto_ray()
