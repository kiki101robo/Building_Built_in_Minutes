import numpy as np
import cv2

def compute_chirality(points_3d, r3, translation):
    # Count points with positive depth relative to the camera
    count_depth = 0
    for point in points_3d:
        # Check if the point is in front of the camera
        if np.dot(r3, (point[:3] - translation)) > 0 and translation[2] > 0:
            count_depth += 1
    return count_depth

def extract_best_pose(rotation_matrices, translation_vectors, points_3d_sets):
    max_count_depth = 0
    best_index = None
    best_rotation = None
    best_translation = None
    best_points_3d = None
    
    # Iterate over all sets of possible poses
    for i in range(len(rotation_matrices)):
        r3 = rotation_matrices[i][2]  # Extract the third row of the rotation matrix
        translation = translation_vectors[i]
        points_3d = points_3d_sets[i]
        
        # Calculate how many points have a positive depth for this pose
        count_depth_positive = compute_chirality(points_3d, r3, translation)
        print(count_depth_positive)  # Debug output to track the depth counts
        
        # Update the best pose if the current one has more points with positive depth
        if count_depth_positive > max_count_depth:
            best_index = i
            max_count_depth = count_depth_positive
            best_rotation = rotation_matrices[best_index]
            best_translation = translation_vectors[best_index]
            best_points_3d = points_3d_sets[best_index]

    return best_rotation, best_translation, best_points_3d, best_index
