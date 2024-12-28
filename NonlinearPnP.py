import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from Plotting import *

def compute_reprojection_error(projection_matrix, image_point, world_point):
    '''
    Computes the reprojection error between the observed image point and the projected point using
    the provided projection matrix.
    '''
    # Convert world point to homogeneous coordinates
    world_point_homogeneous = np.append(world_point.reshape((3, 1)), 1)

    # Project world point onto the image plane
    projected_point = np.dot(projection_matrix, world_point_homogeneous)

    # Convert to inhomogeneous coordinates (normalize by the third coordinate)
    projected_point[:2] /= projected_point[2]

    # Compute squared reprojection error
    reprojection_error = (image_point[0] - projected_point[0])**2 + (image_point[1] - projected_point[1])**2
    return reprojection_error

def optimize_camera_parameters(initial_params, calibration_matrix, image_points, world_points):
    '''
    Optimizes camera parameters to minimize the total reprojection error across all point correspondences.
    '''
    reprojection_errors = []
    rotation = to_euler(initial_params[:4])
    translation = initial_params[4:]

    projection_matrix = projection_matrix(calibration_matrix, rotation, translation)

    for image_point, world_point in zip(image_points, world_points):
        error = compute_reprojection_error(projection_matrix, image_point, world_point)
        reprojection_errors.append(error)

    return np.array(reprojection_errors)

def perform_nonlinear_pnp(calibration_matrix, initial_rotation, initial_translation, image_points, world_points):
    '''
    Refines the camera pose (rotation and translation) using a non-linear Perspective-n-Point approach by
    minimizing the reprojection error.
    '''
    initial_translation = initial_translation.reshape((3, 1))
    initial_quaternion = to_quaternion(initial_rotation)
    initial_parameters = np.append(initial_quaternion, initial_translation)

    result = least_squares(fun=optimize_camera_parameters, x0=initial_parameters, args=(calibration_matrix, image_points, world_points), ftol=1e-10)
    optimized_parameters = result.x

    optimized_rotation = to_euler(optimized_parameters[:4])
    optimized_translation = optimized_parameters[4:].reshape((3, 1))

    return optimized_rotation, optimized_translation
