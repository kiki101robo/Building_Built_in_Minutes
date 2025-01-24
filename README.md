# Building_Built_in_Minutes

The detailed explanation of this project is given in [Group_15___Building_Built_in_Minutes.pdf](https://github.com/user-attachments/files/18268007/Group_15___Building_Built_in_Minutes.pdf).


# Structure from Motion (SfM) - Building Reconstruction

## Project Overview
This project focuses on reconstructing a 3D scene from multiple 2D images using advanced Structure from Motion (SfM) techniques. The methodology involves estimating camera poses and creating a 3D structure through various stages of computer vision algorithms.

### Author
Chahat Deep Singh  
For queries or corrections, please email: chahat[at]terpmail[dot]umd[dot]edu

## Methodology

### Feature Matching
- **Keypoint Detection**: Utilize SIFT to detect distinctive features in each image.
- **Feature Matching**: Implement feature matching across images to find correspondences.
- **Outlier Rejection**: Use RANSAC to reject outliers and refine feature matching.

### Fundamental Matrix Estimation
- **Epipolar Geometry**: Explore the geometry between two views to understand the intrinsic projective properties that relate corresponding points across images.
- **Matrix Calculation**: Estimate the fundamental matrix that encapsulates this geometry.

### Essential Matrix Estimation
- **From Fundamental to Essential**: Convert the fundamental matrix to the essential matrix using camera intrinsic parameters.
- **Camera Pose**: Derive the camera poses (rotation and translation) relative to the scene from the essential matrix.

### Camera Pose Estimation
- **Pose Configurations**: Calculate multiple potential camera poses.
- **Cheirality Check**: Determine the correct camera pose using the cheirality condition to ensure points are in front of the camera.

### Triangulation
- **3D Point Reconstruction**: Triangulate 3D points from camera poses using linear least squares.
- **Non-Linear Refinement**: Refine these points to minimize reprojection errors using non-linear optimization.

### Bundle Adjustment
- **Simultaneous Refinement**: Refine camera poses and 3D points together by minimizing reprojection errors across all images.
- **Visibility Matrix**: Construct a matrix to represent visibility of 3D points from each camera position.

## Implementation Details
- **Pipeline Script**: The entire process is integrated into a single script `Wrapper.py`, which calls modules designed for each step of the SfM pipeline.
- **Modular Design**: Each stage of the pipeline (feature matching, matrix estimations, pose estimations, triangulation, and bundle adjustment) is implemented in separate scripts to facilitate testing and maintenance.

### Scripts
- `GetInliersRANSAC.py`
- `EstimateFundamentalMatrix.py`
- `EssentialMatrixFromFundamentalMatrix.py`
- `ExtractCameraPose.py`
- `LinearTriangulation.py`
- `DisambiguateCameraPose.py`
- `NonlinearTriangulation.py`
- `PnPRANSAC.py`
- `NonlinearPnP.py`
- `BuildVisibilityMatrix.py`
- `BundleAdjustment.py`
- `Wrapper.py` (main script to run the SfM pipeline)

## Further Reading
Refer to the project report for an in-depth analysis of the results and comparisons with other SfM implementations.

##Results
  Inlier Outlier rejection: 
  ![image](https://github.com/user-attachments/assets/15e28c93-cf8c-499f-83da-4d9550953a1e)

  Linear and Non Linear Triangulation: 
  ![image](https://github.com/user-attachments/assets/6bfe0bc1-c0cf-4f88-b9d8-862fc3d3ceb3)
  
  ![image](https://github.com/user-attachments/assets/874b5c05-4720-409f-9bf4-ed93c76fb5dc)

  2D Plot of Linear and Non Linear: 
  ![image](https://github.com/user-attachments/assets/0f83c004-c9df-4338-b6a7-cbbfb8b2997e)
  ![image](https://github.com/user-attachments/assets/1c940e09-bb74-4b63-a25b-8c21aa213f09)

  Visualization after bundle adjustment: 
  ![image](https://github.com/user-attachments/assets/e9ca70ad-bac9-42af-9a0e-3493ecac2cf7)



