import numpy as np


def sample_matrix():
    A = np.zeros((3, 3))
    A[0, 1] = np.random.randn()
    A[0, 2] = np.random.randn()
    A[1, 2] = np.random.randn()

    A[1, 0] = -A[0, 1]
    A[2, 0] = -A[0, 2]
    A[2, 1] = -A[1, 2]

    return A

#def invariant(A):
#    return A[0, 0]*A[1, 1] + A[1, 1]*A[2, 2] + A[0, 0]*A[2, 2] - A[0, 1]*A[1, 0] - A[0, 2]*A[2, 0] - A[1, 2]*A[2, 1]

def invariant(A):
    return A[0]*A[4] + A[4]*A[8] + A[0]*A[8] - A[1]*A[3] - A[2]*A[6] - A[5]*A[7]


def random_orthogonal_matrix(n):
    # Generate a random matrix
    H = np.random.randn(n, n)
    # QR decomposition
    Q, R = np.linalg.qr(H)
    # Make Q uniform by adjusting sign
    D = np.diag(np.sign(np.diag(R)))
    Q = Q @ D
    return Q

# Invariant 2
#def determinant(A):
#    A = A.reshape(2, 2)
#    return np.linalg.det(A)

def data_point():
    
    A = sample_matrix()
    M = random_orthogonal_matrix(3)
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