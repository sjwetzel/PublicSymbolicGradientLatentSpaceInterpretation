import numpy as np


def sample_matrix():
    return np.random.rand(2, 2)

# trace
def invariant(A):
    A = A.reshape(2, 2)
    return np.trace(A)

# Invariant 2
#def determinant(A):
#    A = A.reshape(2, 2)
#    return np.linalg.det(A)

def data_point():
    
    A = sample_matrix()
    M = sample_matrix()
    Minv = np.linalg.inv(M)
    
    A_new = M @ A @ Minv

    A_flat = A.flatten()
    A_new_flat = A_new.flatten()
 
    return np.array([A_flat, A_new_flat])

# Can optimize with numpy     
def triplet_data(size):

    X = []

    for _ in range(size):
        ap_pair = data_point()
        n = sample_matrix().flatten()

        triplet = np.append(ap_pair, n)

        X.append(triplet)
    
    return np.array(X)