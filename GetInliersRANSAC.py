from EstimateFundamentalMatrix import *
import numpy as np
import cv2

def ransac_algorithm(points_img1, points_img2):
    '''
    Implements the RANSAC algorithm to estimate a robust fundamental matrix between two sets of points.
    Points that are consistent with the estimated model are considered inliers.
    '''
    num_points = points_img1.shape[0]
    max_iterations = 10000
    distance_threshold = 0.05
    homogeneous_points_img1 = np.hstack((points_img1, np.ones((points_img1.shape[0], 1))))
    homogeneous_points_img2 = np.hstack((points_img2, np.ones((points_img2.shape[0], 1))))

    max_inliers_count = 0

    for _ in range(max_iterations):
        # Select 8 random points
        random_indices = np.random.choice(num_points, size=8, replace=False)

        # Estimate the fundamental matrix from these points
        selected_points_img1 = points_img1[random_indices, :]
        selected_points_img2 = points_img2[random_indices, :]
        estimated_F = estimate_fundamental_matrix(selected_points_img1, selected_points_img2)

        # Calculate the geometric error for all points
        error = np.abs(np.diag(np.dot(np.dot(homogeneous_points_img2, estimated_F), homogeneous_points_img1.T)))

        # Determine inliers based on the error threshold
        inliers = np.where(error < distance_threshold)[0]
        outliers = np.where(error >= distance_threshold)[0]

        # Update the best model if the current one has more inliers
        if len(inliers) > max_inliers_count:
            max_inliers_count = len(inliers)
            best_inliers_index = inliers
            best_outliers_index = outliers
            best_F = estimated_F

    # Refine the fundamental matrix using all inliers
    refined_points_img1 = points_img1[best_inliers_index]
    refined_points_img2 = points_img2[best_inliers_index]
    refined_F = estimate_fundamental_matrix(refined_points_img1, refined_points_img2)

    return refined_points_img1, refined_points_img2, refined_F
