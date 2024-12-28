import numpy as np
import cv2
import glob

def normalize_points(points):
    '''
    Normalize the input points by translating and scaling them so that their centroid is at the origin
    and their average distance from the origin is sqrt(2).
    This normalization improves the numerical stability of subsequent calculations.
    Reference: https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/arao83/index.html
    '''
    mean_coords = np.mean(points, axis=0)
    sd_u = 1 / np.std(points[:, 0])
    sd_v = 1 / np.std(points[:, 1])

    scale_matrix = np.array([[sd_u, 0, 0], [0, sd_v, 0], [0, 0, 1]])
    translation_matrix = np.array([[1, 0, -mean_coords[0]], [0, 1, -mean_coords[1]], [0, 0, 1]])
    normalization_matrix = np.dot(scale_matrix, translation_matrix)

    homogeneous_points = np.column_stack((points, np.ones(len(points))))
    normalized_points = (np.dot(normalization_matrix, homogeneous_points.T)).T

    return normalized_points, normalization_matrix

def estimate_fundamental_matrix(points_img1, points_img2):
    '''
    Estimate the fundamental matrix using the normalized 8-point algorithm.
    The points from the two images are normalized, and the singular value decomposition (SVD) is used to find
    the matrix that best describes the epipolar constraints between the points.
    '''
    # Normalization of points (if uncommented, normalize both sets of points)
    # points_img1, T1 = normalize_points(points_img1)
    # points_img2, T2 = normalize_points(points_img2)

    x1 = points_img1[:, 0]
    y1 = points_img1[:, 1]
    x2 = points_img2[:, 0]
    y2 = points_img2[:, 1]

    # Assemble matrix A used in the 8-point algorithm
    A = np.zeros((len(x1), 9))
    for i in range(len(x1)):
        A[i] = [x2[i] * x1[i], x2[i] * y1[i], x2[i], y2[i] * x1[i], y2[i] * y1[i], y2[i], x1[i], y1[i], 1]

    # Compute the SVD of A, and extract the fundamental matrix from the last row of V^T
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    F_estimated = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint on the estimated fundamental matrix
    Uf, Sf, Vf = np.linalg.svd(F_estimated, full_matrices=True)
    Sf[2] = 0  # Set the smallest singular value to zero
    F = np.dot(Uf, np.dot(np.diag(Sf), Vf))

    return F
