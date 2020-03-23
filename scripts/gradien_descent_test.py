import numpy as np

matr = np.array([[ 1, 0,-1, 0],
                 [ 0, 1, 0,-1],
                 [-1, 0, 1, 0],
                 [ 0,-1, 0, 1]])

eig_vals, eig_vecs = np.linalg.eig(matr)
print('eig', np.round(eig_vals, 2))

