"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion
import sys
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm


if __name__ == "__main__":
	# ======================================================================================================== #
	# (Optional) This is an example of how to compute the 3D bounds
	# in world coordinates of the convex hull of all camera view
	# frustums in the dataset
	# ======================================================================================================== #
	print("Estimating voxel volume bounds...")

	recon_root = sys.argv[1]
	disp_arr = np.load(os.path.join(recon_root, 'disps.npy'))
	img_arr = np.load(os.path.join(recon_root, 'images.npy'))
	intrinsics_arr = np.load(os.path.join(recon_root, 'intrinsics.npy')) * 8.0  # Droid-SLAM applies x8 multiplication
	poses_arr = np.load(os.path.join(recon_root, 'poses_mtx.npy'))
	masks_arr = np.load(os.path.join(recon_root, 'masks.npy'))

	n_imgs = disp_arr.shape[0]
	cam_intr = np.array([[intrinsics_arr[0, 0], 0., intrinsics_arr[0, 2]], [0., intrinsics_arr[0, 1], intrinsics_arr[0, 3]], [0., 0., 1.]])
	vol_bnds = np.zeros((3,2))
	disp_thres = 0.5  # Tunable threshold

	ht, wd = disp_arr[0].shape
	y, x = np.meshgrid(np.arange(ht).astype(float), np.arange(wd).astype(float))

	# Get zoedepth
	device = "cuda" if torch.cuda.is_available() else "cpu"
	repo = "isl-org/ZoeDepth"
	model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
	zoe = model_zoe_n.to(device)
	depth_zoe_list = []

	for i in tqdm(range(n_imgs)):
		# Read depth image and camera pose
		disp_im = disp_arr[i]
		valid_disp = (disp_im > disp_thres) & masks_arr[i]
		depth_im = np.zeros_like(disp_im)
		depth_im[valid_disp] = 1.0 / disp_im[valid_disp]
		cam_pose = poses_arr[i]  # 4x4 rigid transformation matrix

		# Get zoedepth predictions
		pil_img = Image.fromarray(np.transpose(img_arr[i], (1, 2, 0)))
		depth_zoe = zoe.infer_pil(pil_img)
		depth_zoe_final = np.zeros_like(depth_zoe)

		# Align depth predictions as in MiDAS
		transform_t_estim = np.median(depth_im[valid_disp])
		transform_s_estim = np.mean(np.abs(depth_im[valid_disp] - transform_t_estim))
		transform_t_zoe = np.median(depth_zoe[valid_disp])
		transform_s_zoe = np.mean(np.abs(depth_zoe[valid_disp] - transform_t_zoe))
		
		depth_zoe_final[valid_disp] = (depth_zoe[valid_disp] - transform_t_zoe) / transform_s_zoe * transform_s_estim + transform_t_estim
		depth_zoe_list.append(depth_zoe_final)

		# Compute camera view frustum and extend convex hull
		view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
		vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
		vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
	# ======================================================================================================== #

	# ======================================================================================================== #
	# Integrate
	# ======================================================================================================== #
	# Initialize voxel volume
	print("Initializing voxel volume...")
	tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.01)

	# Loop through RGB-D images and fuse them together
	t0_elapse = time.time()
	for i in range(n_imgs):
		print("Fusing frame %d/%d"%(i+1, n_imgs))

		# Read RGB-D image and camera pose
		color_image = cv2.cvtColor(np.transpose(img_arr[i], [1, 2, 0]), cv2.COLOR_BGR2RGB)
		depth_im = depth_zoe_list[i]
		cam_pose = poses_arr[i]  # 4x4 rigid transformation matrix

		# Integrate observation into voxel volume (assume color aligned with depth)
		tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

	fps = n_imgs / (time.time() - t0_elapse)
	print("Average FPS: {:.2f}".format(fps))

	# Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
	print("Saving mesh to mesh.ply...")
	verts, faces, norms, colors = tsdf_vol.get_mesh()
	fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

	# Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
	print("Saving point cloud to pc.ply...")
	point_cloud = tsdf_vol.get_point_cloud()
	fusion.pcwrite("pc.ply", point_cloud)