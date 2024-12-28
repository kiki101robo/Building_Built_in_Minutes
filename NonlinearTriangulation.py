import numpy as np
import cv2
import scipy.optimize

def non_linear_triangulation(calibration_matrix, rotation1, translation1, rotation2, translation2, image_points1, image_points2, world_points):
    '''
    Optimizes the world coordinates of points using non-linear triangulation methods to minimize the reprojection error.
    '''
    print("sf", rotation1.shape)
    print("sf", rotation2.shape)
    rotation1 = rotation1.reshape((3, 3))
    translation1 = translation1.reshape((3, 1))
    rotation2 = rotation2.reshape((3, 3))
    translation2 = translation2.reshape((3, 1))
    identity_matrix = np.identity(3)

    # Calculate the projection matrices for both camera configurations
    projection_matrix1 = np.dot(calibration_matrix, np.dot(rotation1, np.hstack((identity_matrix, -translation1))))
    projection_matrix2 = np.dot(calibration_matrix, np.dot(rotation2, np.hstack((identity_matrix, -translation2))))

    optimized_world_points = []
    for i in range(len(world_points)):
        optimization_result = scipy.optimize.least_squares(fun=reprojection_loss, x0=world_points[i], args=(image_points1[i], image_points2[i], projection_matrix1, projection_matrix2))
        optimized_world_points.append(optimization_result.x)

    return np.array(optimized_world_points)

def calculate_mean_reprojection_error(calibration_matrix, rotation1, translation1, rotation2, translation2, image_points1, image_points2, world_points):
    '''
    Calculates the mean reprojection error for a given set of world points and their corresponding image points using given camera parameters.
    '''
    rotation1 = rotation1.reshape((3, 3))
    translation1 = translation1.reshape((3, 1))
    rotation2 = rotation2.reshape((3, 3))
    translation2 = translation2.reshape((3, 1))
    identity_matrix = np.identity(3)
    projection_matrix1 = np.dot(calibration_matrix, np.dot(rotation1, np.hstack((identity_matrix, -translation1))))
    projection_matrix2 = np.dot(calibration_matrix, np.dot(rotation2, np.hstack((identity_matrix, -translation2))))

    errors = [reprojection_loss(world_point, image_points1[i], image_points2[i], projection_matrix1, projection_matrix2) for i, world_point in enumerate(world_points)]
    return np.mean(errors)

def reprojection_loss(world_point, image_point1, image_point2, projection_matrix1, projection_matrix2):
    '''
    Calculates the reprojection error for given world points against their corresponding image points in two camera views.
    '''
    projected_point1 = np.dot(projection_matrix1, np.append(world_point, 1))
    projected_point2 = np.dot(projection_matrix2, np.append(world_point, 1))

    u1, v1 = image_point1
    u2, v2 = image_point2

    u1_projected = projected_point1[0] / projected_point1[2]
    v1_projected = projected_point1[1] / projected_point1[2]
    u2_projected = projected_point2[0] / projected_point2[2]
    v2_projected = projected_point2[1] / projected_point2[2]

    error1 = (u1 - u1_projected)**2 + (v1 - v1_projected)**2
    error2 = (u2 - u2_projected)**2 + (v2 - v2_projected)**2

    return error1 + error2

