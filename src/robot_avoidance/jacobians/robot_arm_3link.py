import numpy as np
from numpy import cos, sin


def _get_jacobian(ll, qq):
    return np.array(
        [
            [
                -ll[0] * sin(qq[0])
                - ll[1] * sin(qq[0] + qq[1])
                - ll[2] * sin(qq[0] + qq[1] + qq[2]),
                -ll[1] * sin(qq[0] + qq[1]) - ll[2] * sin(qq[0] + qq[1] + qq[2]),
                -ll[2] * sin(qq[0] + qq[1] + qq[2]),
            ],
            [
                ll[0] * cos(qq[0])
                + ll[1] * cos(qq[0] + qq[1])
                + ll[2] * cos(qq[0] + qq[1] + qq[2]),
                ll[1] * cos(qq[0] + qq[1]) + ll[2] * cos(qq[0] + qq[1] + qq[2]),
                ll[2] * cos(qq[0] + qq[1] + qq[2]),
            ],
            [1, 1, 1],
        ]
    )
