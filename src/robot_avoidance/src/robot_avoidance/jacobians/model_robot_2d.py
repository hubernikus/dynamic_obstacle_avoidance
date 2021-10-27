import numpy as np
from numpy import cos, sin


def _get_jacobian(ll, qq):
    return np.array(
        [
            [
                -ll[0] * sin(qq[0])
                - ll[1] * sin(qq[0] + qq[1])
                - ll[2] * sin(qq[0] + qq[1] + qq[2])
                - ll[3] * sin(qq[0] + qq[1] + qq[2] + qq[3]),
                -ll[1] * sin(qq[0] + qq[1])
                - ll[2] * sin(qq[0] + qq[1] + qq[2])
                - ll[3] * sin(qq[0] + qq[1] + qq[2] + qq[3]),
                -ll[2] * sin(qq[0] + qq[1] + qq[2])
                - ll[3] * sin(qq[0] + qq[1] + qq[2] + qq[3]),
                -ll[3] * sin(qq[0] + qq[1] + qq[2] + qq[3]),
            ],
            [
                -(
                    ll[1] * sin(qq[1])
                    + ll[2] * sin(qq[1] + qq[2])
                    + ll[3] * sin(qq[1] + qq[2] + qq[3])
                )
                * sin(qq[0])
                + (
                    ll[0]
                    + ll[1] * cos(qq[1])
                    + ll[2] * cos(qq[1] + qq[2])
                    + ll[3] * cos(qq[1] + qq[2] + qq[3])
                )
                * cos(qq[0]),
                -(
                    ll[1] * sin(qq[1])
                    + ll[2] * sin(qq[1] + qq[2])
                    + ll[3] * sin(qq[1] + qq[2] + qq[3])
                )
                * sin(qq[0])
                + (
                    ll[1] * cos(qq[1])
                    + ll[2] * cos(qq[1] + qq[2])
                    + ll[3] * cos(qq[1] + qq[2] + qq[3])
                )
                * cos(qq[0]),
                -(ll[2] * sin(qq[1] + qq[2]) + ll[3] * sin(qq[1] + qq[2] + qq[3]))
                * sin(qq[0])
                + (ll[2] * cos(qq[1] + qq[2]) + ll[3] * cos(qq[1] + qq[2] + qq[3]))
                * cos(qq[0]),
                ll[3] * cos(qq[0] + qq[1] + qq[2] + qq[3]),
            ],
            [1, 1, 1, 1],
        ]
    )
