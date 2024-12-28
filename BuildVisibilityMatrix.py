from scipy.sparse import lil_matrix
import numpy as np

def create_sparse_matrix(num_cameras, num_points, cam_indices, pt_indices):
    # Number of measurements: each camera-point pair contributes two entries (x and y pixel coordinates)
    num_measurements = cam_indices.size * 2
    # Number of columns: for camera parameters (7 per camera) and 3D point coordinates (3 per point)
    num_columns = num_cameras * 7 + num_points * 3
    # Creating a sparse matrix with given dimensions and integer type
    sparse_mat = lil_matrix((num_measurements, num_columns), dtype=int)

    # Array of indices corresponding to camera-point pairs
    index_array = np.arange(cam_indices.size)

    # Filling the camera parameter blocks
    for s in range(7):  # There are 7 parameters for each camera
        # Setting the block for x coordinates
        sparse_mat[2 * index_array, cam_indices * 7 + s] = 1
        # Setting the block for y coordinates
        sparse_mat[2 * index_array + 1, cam_indices * 7 + s] = 1

    # Filling the point coordinate blocks
    for s in range(3):  # Each point has 3 coordinates (x, y, z)
        # Adjusting columns for camera parameters and adding point indices for x coordinates
        sparse_mat[2 * index_array, num_cameras * 7 + pt_indices * 3 + s] = 1
        # Same for y coordinates
        sparse_mat[2 * index_array + 1, num_cameras * 7 + pt_indices * 3 + s] = 1

    return sparse_mat
