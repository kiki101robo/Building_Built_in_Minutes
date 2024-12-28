import numpy as np
import cv2

def perform_point_triangulation(calibration_matrix, points_img1, points_img2, rotation_matrix1, camera_center1, rotation_matrix2, camera_center2):
    '''
    Triangulates points from two views given the camera intrinsics and extrinsics.
    '''
    points_3d = []
    identity_matrix = np.identity(3)
    camera_center1 = camera_center1.reshape(3, 1)
    camera_center2 = camera_center2.reshape(3, 1)

    # Compute the projection matrices for both cameras
    projection_matrix1 = np.dot(calibration_matrix, np.dot(rotation_matrix1, np.hstack((identity_matrix, -camera_center1))))
    projection_matrix2 = np.dot(calibration_matrix, np.dot(rotation_matrix2, np.hstack((identity_matrix, -camera_center2))))

    # Convert image points to homogeneous coordinates
    homogeneous_points_img1 = np.hstack((points_img1, np.ones((len(points_img1), 1))))
    homogeneous_points_img2 = np.hstack((points_img2, np.ones((len(points_img2), 1))))

    for i in range(len(homogeneous_points_img1)):
        A = []
        x, y = homogeneous_points_img1[i, :2]
        x_prime, y_prime = homogeneous_points_img2[i, :2]

        # Construct the constraints matrix from the image correspondences
        A.append((y * projection_matrix1[2]) - projection_matrix1[1])
        A.append((x * projection_matrix1[2]) - projection_matrix1[0])
        A.append((y_prime * projection_matrix2[2]) - projection_matrix2[1])
        A.append((x_prime * projection_matrix2[2]) - projection_matrix2[0])

        A = np.array(A)

        # Solve the homogenous equation system using SVD
        _, _, VT = np.linalg.svd(A)
        point_3d_homogeneous = VT[-1]
        point_3d_homogeneous /= point_3d_homogeneous[-1]  # Normalize to make the last coordinate 1

        points_3d.append(point_3d_homogeneous)

    return np.array(points_3d)

def perform_linear_triangulation(rotation_set, translation_set, points_img1, points_img2, calibration_matrix):
    '''
    Performs linear triangulation on multiple rotation and translation pairs to generate 3D points.
    '''
    identity_matrix = np.identity(3)
    zero_translation = np.zeros((3, 1))
    points_3d_sets = []

    for rotation_matrix, translation_vector in zip(rotation_set, translation_set):
        points_3d = perform_point_triangulation(calibration_matrix, points_img1, points_img2, identity_matrix, zero_translation, rotation_matrix, translation_vector)
        points_3d_sets.append(points_3d)

    return points_3d_sets
