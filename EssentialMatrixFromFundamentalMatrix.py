import numpy as np
import cv2

def estimate_essential_matrix(calibration_matrix, fundamental_matrix):
    # Calculate the essential matrix from the fundamental matrix using the calibration matrix
    estimated_E = np.dot(calibration_matrix.T, np.dot(fundamental_matrix, calibration_matrix))
    
    # Decompose the estimated essential matrix to enforce the internal constraints (singular values 1, 1, 0)
    U, singular_values, V = np.linalg.svd(estimated_E, full_matrices=True)
    corrected_singular_values = np.diag(singular_values)
    corrected_singular_values[0, 0], corrected_singular_values[1, 1], corrected_singular_values[2, 2] = 1, 1, 0
    
    # Reconstruct the essential matrix with the corrected singular values
    essential_matrix = np.dot(U, np.dot(corrected_singular_values, V))
    
    return essential_matrix
