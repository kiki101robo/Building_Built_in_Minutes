from LinearPnP import *
import numpy as np
import random

def compute_reprojection_error(image_points, projection_matrix, world_points):
    '''
    Calculates the reprojection error between the projected points and the observed image points.
    '''
    homogeneous_world_points = np.hstack((world_points, np.ones((world_points.shape[0], 1))))
    projected_points = np.dot(projection_matrix, homogeneous_world_points.T).T
    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]

    # Compute the squared error
    errors = np.sum((image_points - projected_points[:, :2]) ** 2, axis=1)
    return errors

def pnp_ransac(world_points, image_points, calibration_matrix):
    '''
    Applies the RANSAC algorithm to find the best camera pose that minimizes reprojection errors
    using the Perspective-n-Point problem solution.
    '''
    if world_points.size == 0 or image_points.size == 0:
        raise ValueError("Input arrays for world or image points are empty.")

    num_points = len(image_points)
    if num_points < 6:
        raise ValueError(f"Not enough points to perform PnP RANSAC (need at least 6, got {num_points}).")

    max_inliers = 0
    threshold = 20
    best_rotation = None
    best_translation = np.zeros(3)
    best_image_points = None
    best_world_points = None

    # Perform RANSAC iterations
    for i in range(1000):
        try:
            random_indices = np.random.choice(num_points, size=6, replace=False)
            selected_world_points = world_points[random_indices]
            selected_image_points = image_points[random_indices]
            rotation, translation = linear_perspective_n_point(selected_world_points, selected_image_points, calibration_matrix)

            # Compute the projection matrix
            translation = translation.reshape((3, 1))
            identity_matrix = np.identity(3)
            projection_matrix = np.dot(calibration_matrix, np.dot(rotation, np.hstack((identity_matrix, -translation))))

            # Calculate reprojection errors for all points
            errors = compute_reprojection_error(image_points, projection_matrix, world_points)
            inliers = np.where(errors < threshold)[0]

            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_image_points = image_points[inliers]
                best_world_points = world_points[inliers]
                best_rotation = rotation
                best_translation = translation
        except Exception as e:
            print(f"An error occurred during RANSAC iteration {i}: {str(e)}")
            continue

    return best_rotation, best_translation, best_image_points, best_world_points
