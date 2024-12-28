import numpy as np

def linear_perspective_n_point(world_points, image_points, calibration_matrix):
    '''
    Solves the Perspective-n-Point problem using a linear approach.
    This method estimates the camera pose (rotation and translation) from 3D-2D point correspondences.
    '''
    A = np.empty((0, 12), np.float32)

    for i in range(len(image_points)):
        # Extract image coordinates
        x, y = image_points[i][0], image_points[i][1]

        # Normalize the image points using the calibration matrix
        normalized_points = np.dot(np.linalg.inv(calibration_matrix), np.array([[x], [y], [1]]))
        normalized_points /= normalized_points[2]

        # Corresponding 3D world coordinates
        world_point = world_points[i].reshape((3, 1))
        homogeneous_world_point = np.append(world_point, 1)  # Convert to homogeneous coordinates

        zeros = np.zeros((4,))

        # Building the matrix A using the 3D-2D correspondences
        A_1 = np.hstack((zeros, -homogeneous_world_point.T, normalized_points[1] * (homogeneous_world_point.T)))
        A_2 = np.hstack((homogeneous_world_point.T, zeros, -normalized_points[0] * (homogeneous_world_point.T)))
        A_3 = np.hstack((-normalized_points[1] * (homogeneous_world_point.T), normalized_points[0] * homogeneous_world_point.T, zeros))

        # Append the rows to matrix A
        for a in [A_1, A_2, A_3]:
            A = np.append(A, [a], axis=0)

    # Decompose matrix A using Singular Value Decomposition
    U, S, VT = np.linalg.svd(A)
    V = VT.T

    # The last column of V contains the flattened pose parameters
    pose = V[:, -1].reshape((3, 4))

    # Extract rotation and translation from the pose matrix
    rotation = pose[:, :3]
    translation = pose[:, 3].reshape((3, 1))

    # Impose orthogonality constraint on rotation matrix
    U, _, VT = np.linalg.svd(rotation)
    rotation = np.dot(U, VT)

    # Ensure the determinant of the rotation matrix is positive
    if np.linalg.det(rotation) < 0:
        rotation = -rotation
        translation = -translation

    # Compute the camera center from the rotation and translation
    camera_center = -np.dot(rotation.T, translation)

    return rotation, camera_center
