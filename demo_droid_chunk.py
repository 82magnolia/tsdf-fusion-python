"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion
import sys
import os
import matplotlib.pyplot as plt


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
	
	if os.path.exists(os.path.join(recon_root, 'scale.txt')):  # Known scale case
		with open(os.path.join(recon_root, 'scale.txt'), 'r') as f:
			scale = float(f.readline())
		disp_thres = 1.0  # Large disparity threshold for known scale
		voxel_size = 0.002
		q_value = 0.05  # Quantile to filter view frustrums
		num_chunk = 4  # Arbitrary number of chunks chosen to bound memory
	else:
		scale = 1.0
		disp_thres = 0.5
		voxel_size = 0.005
		q_value = 0.0  # Quantile to filter view frustrums
		num_chunk = 1

	n_imgs = disp_arr.shape[0]
	cam_intr = np.array([[intrinsics_arr[0, 0], 0., intrinsics_arr[0, 2]], [0., intrinsics_arr[0, 1], intrinsics_arr[0, 3]], [0., 0., 1.]])
	vol_bnds = np.zeros((3,2))

	ht, wd = disp_arr[0].shape
	y, x = np.meshgrid(np.arange(ht).astype(float), np.arange(wd).astype(float))
	
	# Set number of images to use per step
	n_imgs_per_step = []
	for c_idx in range(num_chunk):
		if c_idx != num_chunk - 1:
			n_imgs_per_step.append(n_imgs // num_chunk)
		else:
			n_imgs_per_step.append(n_imgs // num_chunk + n_imgs % num_chunk)

	# Make reconstructions per step
	total_verts = []
	total_faces = []
	total_norms = []
	total_colors = []
	total_point_clouds = []
	for c_idx in range(num_chunk):
		print(f"Processing chunk {c_idx}...")
		total_frust_pts = []
		if c_idx == 0:
			start_img_idx = 0
			end_img_idx = start_img_idx + n_imgs_per_step[c_idx]
			face_start_idx = 0
		else:
			start_img_idx = end_img_idx
			end_img_idx = start_img_idx + n_imgs_per_step[c_idx]
		for i in range(start_img_idx, end_img_idx):
			# Read depth image and camera pose
			disp_im = disp_arr[i]
			depth_im = np.zeros_like(disp_im)
			depth_im[(disp_im > disp_thres) & masks_arr[i]] = 1.0 / disp_im[(disp_im > disp_thres) & masks_arr[i]]
			cam_pose = poses_arr[i]  # 4x4 rigid transformation matrix

			# Compute camera view frustum and extend convex hull
			view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
			total_frust_pts.append(view_frust_pts.T)
		# ======================================================================================================== #
		total_frust_pts = np.concatenate(total_frust_pts, axis=0)  # (N, 3)
		vol_bnds_min = np.quantile(total_frust_pts, q=q_value, axis=0)
		vol_bnds_max = np.quantile(total_frust_pts, q=1 - q_value, axis=0)
		vol_bnds = np.stack([vol_bnds_min, vol_bnds_max], axis=0).T
		# ======================================================================================================== #
		# Integrate
		# ======================================================================================================== #
		# Initialize voxel volume
		print("Initializing voxel volume...")
		tsdf_vol = fusion.TSDFVolume(vol_bnds, trunc_margin=5, voxel_size=voxel_size)

		# Loop through RGB-D images and fuse them together
		t0_elapse = time.time()
		for i in range(start_img_idx, end_img_idx):
			print("Fusing frame %d/%d"%(i+1 - start_img_idx, n_imgs_per_step[c_idx]))

			# Read RGB-D image and camera pose
			color_image = cv2.cvtColor(np.transpose(img_arr[i], [1, 2, 0]), cv2.COLOR_BGR2RGB)
			disp_im = disp_arr[i]
			depth_im = np.zeros_like(disp_im)
			depth_im[(disp_im > disp_thres) & masks_arr[i]] = 1.0 / disp_im[(disp_im > disp_thres) & masks_arr[i]]
			cam_pose = poses_arr[i]  # 4x4 rigid transformation matrix

			# Integrate observation into voxel volume (assume color aligned with depth)
			tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

		fps = n_imgs_per_step[c_idx] / (time.time() - t0_elapse)
		print("Average FPS: {:.2f}".format(fps))

		# Get mesh from voxel volume
		print("Creating mesh...")
		verts, faces, norms, colors = tsdf_vol.get_mesh()
		
		# Update face start index to make consistent mesh
		faces += face_start_idx
		face_start_idx += verts.shape[0]
		verts = verts / scale

		# Get point cloud from voxel volume
		print("Creating point cloud...")
		point_cloud = tsdf_vol.get_point_cloud()
		point_cloud[:, :3] = point_cloud[:, :3] / scale

		total_verts.append(verts)
		total_faces.append(faces)
		total_norms.append(norms)
		total_colors.append(colors)
		total_point_clouds.append(point_cloud)

		# Free memory
		tsdf_vol._tsdf_vol_gpu.free()
		tsdf_vol._weight_vol_gpu.free()
		tsdf_vol._color_vol_gpu.free()
	
	total_verts = np.concatenate(total_verts, axis=0)
	total_faces = np.concatenate(total_faces, axis=0)
	total_norms = np.concatenate(total_norms, axis=0)
	total_colors = np.concatenate(total_colors, axis=0)
	total_point_clouds = np.concatenate(total_point_clouds, axis=0)

	# Save point cloud
	fusion.meshwrite("mesh.ply", total_verts, total_faces, total_norms, total_colors)	
	fusion.pcwrite("pc.ply", total_point_clouds)
