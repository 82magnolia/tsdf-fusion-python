"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion
import sys
import os
import matplotlib.pyplot as plt
import open3d as o3d
import torch


def make_img(xyz, rgb, cam_intrinsic, resolution, return_torch=False, point_pad_size=1, resize_rate=1):
    # make image from (N, 3) arrray xyz and (N, 3) array rgb
    with torch.no_grad():
        dist = torch.norm(xyz, dim=-1)
        mod_idx = torch.argsort(dist)
        mod_idx = torch.flip(mod_idx, dims=[0])
        mod_xyz = xyz.clone().detach()[mod_idx]
        mod_rgb = rgb.clone().detach()[mod_idx]
        proj_resolution = (resolution[0] // resize_rate, resolution[1] // resize_rate)

        proj_cam_intrinsic = cam_intrinsic / resize_rate
        proj_xyz = mod_xyz[:, :2]
        proj_xyz /= mod_xyz[:, -1:]
        proj_xyz = torch.cat([proj_xyz, torch.ones_like(proj_xyz[:, 0:1])], dim=-1)
        proj_coord = (proj_cam_intrinsic @ proj_xyz.T).T
        hor_coord = proj_coord[:, 0]
        ver_coord = proj_coord[:, 1]
        inlier_idx = (hor_coord >= 0) & (hor_coord <= proj_resolution[1]) & (ver_coord >= 0) & (ver_coord <= proj_resolution[0])
        coord_idx = proj_coord[inlier_idx, :2].long()
        coord_idx = torch.flip(coord_idx, [-1])
        coord_idx = tuple(coord_idx.T)
        mod_rgb = mod_rgb[inlier_idx].float()

        image = torch.ones([proj_resolution[0], proj_resolution[1], 3], dtype=torch.float).to(xyz.device)

        if len(coord_idx[0]) == 0:
            if not return_torch:
                image = image.cpu().numpy().astype(np.uint8)
            return image

        temp = torch.ones_like(coord_idx[0]).to(xyz.device)
        coord_idx_list = []
        pad_1d_list = [i for i in range(point_pad_size + 1)] + [-i for i in range(1, point_pad_size + 1)]
        pad_1d_hor, pad_1d_ver = np.meshgrid(pad_1d_list, pad_1d_list)
        pad_1d_hor = pad_1d_hor.flatten().tolist()
        pad_1d_ver = pad_1d_ver.flatten().tolist()

        for pad_hor, pad_ver in zip(pad_1d_hor, pad_1d_ver):
            pad_coord_idx = (torch.clamp(coord_idx[0] + pad_hor * temp, max=proj_resolution[0] - 1), 
                torch.clamp(coord_idx[1] + pad_ver * temp, max=proj_resolution[1] - 1))
            coord_idx_list.append(pad_coord_idx)

        for idx in range(len(pad_1d_hor)):
            image.index_put_(coord_idx_list[idx], mod_rgb, accumulate=False)

        image = image * 255

        if not return_torch:
            image = image.cpu().numpy().astype(np.uint8)

    return image


if __name__ == "__main__":
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    print("Rendering views at estimated poses...")
    recon_root = sys.argv[1]
    pcd_name = sys.argv[2]
    render_root = sys.argv[3]
    img_root = sys.argv[4]
    calib_file = sys.argv[5]

    if not os.path.exists(render_root):
        os.makedirs(render_root)

    disp_arr = np.load(os.path.join(recon_root, 'disps.npy'))
    img_arr = np.load(os.path.join(recon_root, 'images.npy'))
    intrinsics_arr = np.load(os.path.join(recon_root, 'intrinsics.npy')) * 8.0  # Droid-SLAM applies x8 multiplication
    poses_arr = np.load(os.path.join(recon_root, 'poses_mtx.npy'))
    masks_arr = np.load(os.path.join(recon_root, 'masks.npy'))
    img_list = open(os.path.join(recon_root, 'images.txt'))
    pcd = o3d.io.read_point_cloud('pc.ply')
    xyz = np.asarray(pcd.points)
    xyz_homo = np.concatenate([xyz, np.ones_like(xyz[:, 0:1])], axis=-1)
    rgb = np.asarray(pcd.colors)

    calib = np.loadtxt(calib_file, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    n_imgs = disp_arr.shape[0]
    cam_intr = K
    img_files = [os.path.join(img_root, f.strip().replace('images/', '')) for f in img_list.readlines()]
    render_res = (cv2.imread(img_files[0]).shape[0], cv2.imread(img_files[0]).shape[1])
    resize_rate = 2
    for i in range(n_imgs):
        # Read depth image and camera pose
        cam_pose = poses_arr[i]  # 4x4 rigid transformation matrix
        transform_xyz_homo = (np.linalg.inv(cam_pose) @ xyz_homo.T).T
        render_img = make_img(torch.from_numpy(transform_xyz_homo[:, :3]), torch.from_numpy(rgb), 
            torch.from_numpy(cam_intr), render_res, point_pad_size=1, resize_rate=resize_rate)
        orig_img = cv2.cvtColor(cv2.imread(img_files[i]), cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, (render_res[1] // resize_rate, render_res[0] // resize_rate))
        plt.imshow(np.concatenate([orig_img, render_img], axis=0))
        plt.show()
    # ======================================================================================================== #
