import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from BuildVisibilityMatrix import *
import time
from Plotting import *

def compute_reprojection_errors(calibration_matrix, img_pts_2d, camera_params, world_pts):
    # Add a column of ones to the 3D world points to make them homogeneous
    homogeneous_world_pts = np.hstack((world_pts, np.ones((world_pts.shape[0], 1))))
    projected_img_pts = np.empty((0, 2), dtype=np.float32)

    # Project each 3D point into 2D using the camera parameters
    for i, p in enumerate(homogeneous_world_pts):
        rotation = to_euler(camera_params[i, :4])  # Convert quaternion to Euler angles
        translation = camera_params[i, 4:]         # Extract translation vector
        projection_mat = projection_matrix(calibration_matrix, rotation, translation)  # Compute projection matrix
        p = p.reshape((4, 1))
        projected_point = np.dot(projection_mat, p)  # Project the point
        projected_point /= projected_point[2]        # Convert from homogeneous to Cartesian coordinates
        projected_img_pts = np.append(projected_img_pts, projected_point[:2].reshape((1, 2)), axis=0)

    # Calculate the reprojection error as the difference between observed and projected points
    reprojection_error = img_pts_2d - projected_img_pts
    return reprojection_error.ravel()

def prepare_optimization(pose_set, world_points, mapping_2d_3d):
    initial_params = np.empty(0, dtype=np.float32)
    point_indices = np.empty(0, dtype=int)
    img_points_2d = np.empty((0, 2), dtype=np.float32)
    camera_indices = np.empty(0, dtype=int)

    num_cameras = max(pose_set.keys())

    # Aggregate initial camera parameters and corresponding 2D-3D point indices
    for k in pose_set.keys():
        quaternion = to_quaternion(pose_set[k][:, 0:3])  # Convert rotation matrix to quaternion
        translation = pose_set[k][:, 3]                  # Extract translation vector
        initial_params = np.append(initial_params, quaternion.reshape(-1))
        initial_params = np.append(initial_params, translation)

        # Collect mapping from 2D points in images to 3D world points
        for p in mapping_2d_3d[k]:
            point_indices = np.append(point_indices, [p[1]])
            img_points_2d = np.append(img_points_2d, [p[0]], axis=0)
            camera_indices = np.append(camera_indices, [k-1])

    initial_params = np.append(initial_params, world_points.flatten())
    num_world_points = world_points.shape[0]

    return num_cameras, num_world_points, point_indices, img_points_2d, camera_indices, initial_params

def reprojection_loss(params, num_cameras, num_world_points, point_indices, img_points_2d, camera_indices, calibration_matrix):
    # Extract camera parameters and world points from the flattened array
    camera_params = params[:num_cameras*7].reshape((num_cameras, 7))
    world_points = params[num_cameras*7:].reshape((-1, 3))
    # Compute reprojection error for the given parameters
    error = compute_reprojection_errors(calibration_matrix, img_points_2d, camera_params[camera_indices], world_points[point_indices])
    return error

def bundle_adjust(pose_set, world_points, mapping_2d_3d, calibration_matrix):
    num_cameras, num_world_points, point_indices, img_points_2d, camera_indices, initial_params = prepare_optimization(pose_set, world_points, mapping_2d_3d)
    
    # Define the sparse matrix structure for efficient Jacobian computation
    sparse_mat = create_sparse_matrix(num_cameras, num_world_points, camera_indices, point_indices)
    start_time = time.time()
    result = least_squares(fun=reprojection_loss, x0=initial_params, jac_sparsity=sparse_mat, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=(num_cameras, num_world_points, point_indices, img_points_2d, camera_indices, calibration_matrix))
    end_time = time.time()
    print("Optimization took -- {} seconds".format(end_time - start_time))

    # Extract optimized camera parameters and world points
    camera_params_opt = result.x[:num_cameras*7].reshape((num_cameras, 7))
    world_points_opt = result.x[num_cameras*7:].reshape((num_world_points, 3))
    optimized_poses = {}
    i = 1
    for cp in camera_params_opt:
        rotation = to_euler(cp[:4])
        translation = cp[4:].reshape((3, 1))
        optimized_poses[i] = np.hstack((rotation, translation))
        i += 1

    return optimized_poses, world_points_opt
