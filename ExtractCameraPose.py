import numpy as np
import cv2

def extract_rotation_translation_sets(essential_matrix):
    '''
    Decompose an essential matrix into possible rotation (R) and translation (T) matrices.
    This function follows the procedure from Hartley and Zisserman's multiple-view geometry.
    It returns four possible sets of (R, T) that need to be further checked for correctness
    through triangulation and chirality conditions.
    '''

    # SVD decomposition of the essential matrix
    U, S, Vt = np.linalg.svd(essential_matrix)

    # Define the W matrix used for constructing R
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Possible rotation matrices from the decomposition
    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W, Vt))
    R3 = np.dot(U, np.dot(W.T, Vt))
    R4 = np.dot(U, np.dot(W.T, Vt))

    # Possible translation vectors (up to scale)
    T1 = U[:, 2]
    T2 = -U[:, 2]
    T3 = U[:, 2]
    T4 = -U[:, 2]

    rotations = [R1, R2, R3, R4]
    translations = [T1, T2, T3, T4]

    # Ensure that each rotation matrix has a determinant of 1
    for i in range(len(rotations)):
        if np.linalg.det(rotations[i]) < 0:
            rotations[i] = -rotations[i]
            translations[i] = -translations[i]

    return rotations, translations
