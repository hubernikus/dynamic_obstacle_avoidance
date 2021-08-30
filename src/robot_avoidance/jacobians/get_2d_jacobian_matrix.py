import numpy as np 
from numpy import cos, sin 

def get_2d_jacobian_matrix(ll, qq):
    return np.array([[0, -ll[2]*sin(qq[1]) + ll[3]*(-sin(qq[1])*cos(qq[2]) - sin(qq[2])*cos(qq[1])), ll[3]*(-sin(qq[1])*cos(qq[2]) - sin(qq[2])*cos(qq[1]))],
[0, ll[2]*cos(qq[1]) + ll[3]*(-sin(qq[1])*sin(qq[2]) + cos(qq[1])*cos(qq[2])), ll[3]*(-sin(qq[1])*sin(qq[2]) + cos(qq[1])*cos(qq[2]))],
[1, 1, 1],
])