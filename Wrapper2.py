import numpy as np
import cv2
import glob
from os.path import exists
import argparse
from distutils.util import strtobool
from scipy.spatial.transform import Rotation 
from GetData import *
from GetInliersRANSAC import *
import matplotlib.pyplot as plt
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from Plotting import *
from PnPRANSAC import *
from NonlinearPnP import *
from BundleAdjustment import *

def main():

    k = np.array([[531.122155322710, 0, 407.192550839899],
                  [0, 531.541737503901, 313.308715048366],
                  [0, 0, 1]])

   
    folder = r"C:\Users\Kirti\Documents\733\P3Data"
    filtered_avail = 0
    images = [cv2.imread(img) for img in sorted(glob.glob(str(folder) + '/*.png'))]
    n_imgs = 5

   

    if filtered_avail:
        fundamental_matrix = np.load('fmatrix.npy', allow_pickle=True)
    else:
        fundamental_matrix = np.zeros(shape=(n_imgs, n_imgs), dtype=object)
        for i in range(n_imgs - 1):
            for j in range(i + 1, n_imgs):

                print("RANSAC for image" + str(i + 1) + " and " + str(j + 1))
                pair_num = str(i + 1) + str(j + 1)
                file_name = "matching" + pair_num + ".txt"
                if exists(folder + "/" + file_name):
                    points1, points2 = get_pts(folder, file_name)
                    point1_fil, point2_fil, F_best = ransac_algorithm(points1, points2)
                    save_file_name = "ransac" + pair_num + ".txt"
                    for idx in range(len(point1_fil)):
                        save_file = open(save_file_name, 'a')
                        save_file.write(str(point1_fil[idx][0]) + " " + str(point1_fil[idx][1]) + " " + str(point2_fil[idx][0]) + " " + str(point2_fil[idx][1]) + "\n")
                        save_file.close()
                        print(idx)

                    fundamental_matrix[i, j] = F_best.reshape((3, 3))

                    display_ransac(images[i], images[j], point1_fil, point2_fil, points1, points2, pair_num)
                else:
                    continue

    # Rest of the pipeline only for 1st two images
    file_name = "matching" + str(12) + ".txt"
    points1, points2 = get_pts(folder, file_name)
    point1_fil, point2_fil, F_best = ransac_algorithm(points1, points2)
    F_matrix = F_best
    print(F_matrix)

    # Estimate Essential Matrix from Fundamental Matrix
    E_matrix = estimate_essential_matrix(k, F_matrix)
    print(E_matrix)

    # Extract Poses of Camera (will be 4)
    R_set, T_set = extract_rotation_translation_sets(E_matrix)

    # Linear Triangulation 
    point3D_set = perform_linear_triangulation(R_set, T_set, point1_fil, point2_fil, k)
    plot_poses(R_set, T_set, point3D_set)
    #plot_poses_3d(R_set, T_set, point3D_set)

    # Get pose of camera using cheirality condition
    R_best, T_best, X_, index = extract_best_pose(R_set, T_set, point3D_set)
    #plot_selectedpose_3d(T_best, R_best, X_, index)

    R1 = np.identity(3)
    T1 = np.zeros((3, 1))

    # Non-Linear Triangulation
    X_nl = non_linear_triangulation(R1, T1, R_best, T_best, point1_fil, point2_fil, X_, k)
    plot_linear_nonlinear(X_, X_nl, index)
    #plot_linear_nonlinear_3d(X_, X_nl, index)

    # Calculate error
    error_prior = calculate_mean_reprojection_error(R1, T1, R_best, T_best, point1_fil, point2_fil, X_, k)
    print("Linear Triangulation", error_prior)
    print("--------------------------------------------------------------------")
    error_post = calculate_mean_reprojection_error(R1, T1, R_best, T_best, point1_fil, point2_fil, X_nl, k)
    print("Non-linear triangulation", error_post)
    print("--------------------------------------------------------------------")

    print("Performing linear PnP to estimate pose of cameras 3-5")
    # Using correspondences between the following image pairs for PnP

    # Create a dict consisting of 2d-3d correspondences of all images
    corresp_2d_3d = {}

    # X_set stores all the 3d points
    X_set = []

    # First we need to get inliers of image i(3-5) wrt previously estimated camera pose so that we
    # match the 2D image point with the already calculated 3D point
    img1_2d_3d = point1_fil
    X_list_refined = np.reshape(X_nl[:, :3], (img1_2d_3d.shape[0], 3))
    img1_2d_3d = np.hstack((img1_2d_3d, X_list_refined))
    corresp_2d_3d[1] = img1_2d_3d

    # Same thing for image 2
    img2_2d_3d = point2_fil
    img2_2d_3d = np.hstack((img2_2d_3d, X_list_refined))
    corresp_2d_3d[2] = img2_2d_3d

    # Add the 3d points to X_set
    X_set.append(X_list_refined)
    X_set = np.array(X_set).reshape((X_list_refined.shape))

    # Map is used for BA. It stores image points and indices of corresp 3d points
    ba_map = {}
    ba_map[1] = zip(corresp_2d_3d[1][:, 0:2], range(X_set.shape[0]))
    ba_map[2] = zip(corresp_2d_3d[2][:, 0:2], range(X_set.shape[0]))

    pose_set = {}
    pose_set_pnp = {}
    T1 = np.zeros(3)
    R1 = np.identity(3)
    pose_set[1] = np.hstack((R1, T1.reshape((3, 1))))
    pose_set[2] = np.hstack((R_best, T_best.reshape((3, 1))))

    pose_set_pnp[1] = np.hstack((R1, T1.reshape((3, 1))))
    pose_set_pnp[2] = np.hstack((R_best, T_best.reshape((3, 1))))

    # Estimate pose for the remaining cams
    for i in range(2, n_imgs):

        ref_img_num = i
        new_img_num = i + 1
        img_pair = str(ref_img_num) + str(new_img_num)
        file_name = "ransac" + img_pair + ".txt"
        print(file_name)
        
        # Construct projection matrix of ref image
        R_ref = pose_set[ref_img_num][:, 0:3].reshape((3, 3))
        print(R_ref)
        C_ref = pose_set[ref_img_num][:, 3].reshape((3, 1))

        # Get the 2d-3d correspondences for the 1st ref image
        ref_img_2d_3d = corresp_2d_3d[ref_img_num]
        ref_2d = ref_img_2d_3d[:, 0:2]
        ref_3d = ref_img_2d_3d[:, 2:]

        # Next we must compare it with the points found using given matches
        points1, points2 = get_ransac_pts(folder, file_name)

        # Obtain the 3D corresp for the new image
        ref_2d, new_2d, new_3d, matches = compute_correspondences(ref_2d, ref_3d, points1, points2)
        print(len(matches))

        # PnP RANSAC
        print("Performing PnP RANSAC to refine the poses")
        R_new_lt, T_new_lt, pnp_2d, pnp_3d = pnp_ransac(new_3d, new_2d, k)
        P = projection_matrix(k, R_new_lt, T_new_lt)
        pnp_error = np.mean(compute_reprojection_error(new_2d, P, new_3d))
        print("Linear PnP error", pnp_error)
        print("--------------------------------------------------------------------")

        # Non-linear PnP
        print("Performing Non-linear PnP to obtain optimal pose")
        R_new, T_new = perform_nonlinear_pnp(k, R_new_lt, T_new_lt, pnp_2d, pnp_3d)

        P = projection_matrix(k, R_new, T_new)
        nonpnp_error = np.mean(compute_reprojection_error(new_2d, P, new_3d))
        print("Non-linear PnP error", nonpnp_error)
        print("--------------------------------------------------------------------")

        # Linear triangulation
        print("Performing Linear Triangulation to obtain 3d equiv for remaining 2d points")
        pt1 = matches[:, 0:2]
        pt2 = matches[:, 2:4]

        # Find the 2d-3d mapping for the remaining image points in the new image by doing triangulation
        X_lin_tri = perform_point_triangulation(k, pt1, pt2, R_ref, C_ref, R_new, T_new)
        lt_error = calculate_mean_reprojection_error(R_ref, C_ref, R_new, T_new, pt1, pt2, X_lin_tri, k)
        print("Linear Triangulation error", lt_error)
        print("--------------------------------------------------------------------")

        colormap = ['r', 'b', 'g', 'y']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        C = T_new
        R = R_new
        X = X_lin_tri
        R = Rotation.from_matrix(R).as_rotvec()
        R1 = np.rad2deg(R)
        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        t._transform = t.get_transform().rotate_deg(int(R1[1]))
        ax = plt.gca()
        ax.scatter((C[0]), (C[2]), marker=t, s=250, color=colormap[index])
        ax.scatter(X[:, 0], X[:, 2], s=4, color=colormap[index], label='cam' + str(index))
        plt.xlim(-20, 20)
        plt.ylim(-20, 30)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.legend()
        plt.savefig("linear_t" + img_pair + ".png")
        # plt.show()

        # Non-Linear triangulation
        print("Performing Non-Linear Triangulation to obtain 3d equiv for remaining 2d points")
        X_new_nl = non_linear_triangulation(R_ref, C_ref, R_new, T_new, pt1, pt2, X_lin_tri, k)
        nlt_error = calculate_mean_reprojection_error(R_ref, C_ref, R_new, T_new, pt1, pt2, X_new_nl, k)
        print("Non-linear Triangulation error", nlt_error)
        print("--------------------------------------------------------------------")

        colormap = ['r', 'b', 'g', 'y']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        C = T_new
        R = R_new
        X = X_new_nl
        R = Rotation.from_matrix(R).as_rotvec()
        R1 = np.rad2deg(R)
        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        t._transform = t.get_transform().rotate_deg(int(R1[1]))
        ax = plt.gca()
        ax.scatter((C[0]), (C[2]), marker=t, s=250, color=colormap[index])
        ax.scatter(X[:, 0], X[:, 2], s=4, color=colormap[index], label='cam' + str(index))
        plt.xlim(-20, 20)
        plt.ylim(-20, 30)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.legend()
        plt.savefig("non_linear_t" + img_pair + ".png")

        # Store the current pose after non linear pnp
        pose_set[new_img_num] = np.hstack((R_new, T_new.reshape((3, 1))))
        pose_set_pnp[new_img_num] = np.hstack((R_new, T_new.reshape((3, 1))))
        pts_img_all = pt2
        X_all = X_new_nl[:, :3]
        corresp_2d_3d[new_img_num] = np.hstack((pts_img_all, X_all))

        P = projection_matrix(k, R_new, T_new)
        draw_projected_points(images[new_img_num - 1], X_all, pts_img_all, P, str(new_img_num))

        # Do bundle adjustment
        index_start = X_set.shape[0]
        index_end = X_set.shape[0] + X_all.shape[0]
        ba_map[new_img_num] = zip(pts_img_all, range(index_start, index_end))
        X_set = np.append(X_set, X_all, axis=0)

        print("Doing Bundle Adjustment --> ")
        pose_set_opt, X_set_opt = bundle_adjust(pose_set, X_set, ba_map, k)
        #plot_poses_3d(pose_set_opt, X_set_opt, X_set_opt)

        # Compute reproj error after BA
        R_ba = pose_set_opt[new_img_num][:, 0:3]
        C_ba = pose_set_opt[new_img_num][:, 3]
        X_all_ba = X_set_opt[index_start:index_end].reshape((X_all.shape[0], 3))

        X_set = X_set_opt

        corresp_2d_3d[new_img_num] = np.hstack((pts_img_all, X_all_ba))
        pose_set = pose_set_opt

        pose = pose_set[new_img_num]
        T_ = pose[:, 3]
        R_ = pose[:, 0:3]
        P = projection_matrix(k, R_, T_)
        ba_error = np.mean(compute_reprojection_errors(pts_img_all, P, X_all_ba))
        print("BA error:", ba_error)

        colormap = ['r', 'b', 'g', 'y', 'c', 'm']
        for i in range(len(pose_set)):
            pose = pose_set[i + 1]
            pt = corresp_2d_3d[i + 1]
            C = pose[:, 3]
            R = pose[:, 0:3]
            X = pt[:, 2:]
            R = Rotation.from_matrix(R).as_rotvec()
            R1 = np.rad2deg(R)
            t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
            t._transform = t.get_transform().rotate_deg(int(R1[1]))
            ax = plt.gca()
            ax.scatter((C[0]), (C[2]), marker=t, s=250, color=colormap[i])
            ax.scatter(X[:, 0], X[:, 2], s=4, color=colormap[i], label='cam' + str(i))
        plt.xlim(-20, 20)
        plt.ylim(-20, 30)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.legend()
        plt.savefig(img_pair + '.png')

        input('q')

        print("......................................")

    colormap = ['y', 'b', 'c', 'm', 'r', 'k']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormap = ['r', 'b', 'g', 'y', 'c', 'm']
    for i in range(len(pose_set)):
        pose = pose_set[i + 1]
        pt = corresp_2d_3d[i + 1]
        C = pose[:, 3]
        R = pose[:, 0:3]
        X = pt[:, 2:]
        R = Rotation.from_matrix(R).as_rotvec()
        R1 = np.rad2deg(R)
        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        t._transform = t.get_transform().rotate_deg(int(R1[1]))
        ax = plt.gca()
        ax.scatter((C[0]), (C[2]), marker=t, s=250, color=colormap[i])
        ax.scatter(X[:, 0], X[:, 2], s=4, color=colormap[i], label='cam' + str(i))
    plt.xlim(-20, 20)
    plt.ylim(-20, 30)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend()
    plt.savefig('BundleAdjustment' + img_pair + '.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormap = ['r', 'b', 'g', 'y', 'c', 'm']
    for i in range(len(pose_set_pnp)):
        pose = pose_set_pnp[i + 1]
        C = pose[:, 3]
        R = pose[:, 0:3]
        X = pt[:, 2:]
        R = Rotation.from_matrix(R).as_rotvec()
        R1 = np.rad2deg(R)
        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        t._transform = t.get_transform().rotate_deg(int(R1[1]))
        ax = plt.gca()
        ax.scatter((C[0]), (C[2]), marker=t, s=250, color=colormap[i])
    plt.xlim(-20, 20)
    plt.ylim(-20, 30)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend()
    plt.savefig('PnP' + img_pair + '.png')

if __name__ == '__main__':
    main()
