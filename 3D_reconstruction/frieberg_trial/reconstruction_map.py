# reconstruct_map.py

import os
import cv2
import numpy as np
from tqdm import tqdm
import open3d as o3d

def load_rgb_images(scene_rgb_dir):
    """Load RGB images from the specified directory."""
    image_files = sorted([
        f for f in os.listdir(scene_rgb_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    rgb_images = []
    for f in image_files:
        img_path = os.path.join(scene_rgb_dir, f)
        img = cv2.imread(img_path)
        if img is not None:
            rgb_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            print(f"Warning: Unable to load image {img_path}")
    return rgb_images, image_files

def load_depth_maps(depth_dir, image_files):
    """Load depth maps corresponding to the RGB images."""
    depth_maps = []
    for f in image_files:
        depth_path = os.path.join(depth_dir, f"{f[:-4]}-dpt_swin2_tiny_256.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is not None:
            # Assuming depth maps are normalized between 0 and 1
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            depth_maps.append(depth)
        else:
            print(f"Warning: Unable to load depth map {depth_path}")
            depth_maps.append(None)
    return depth_maps

def extract_features_and_matches(img1, img2):
    """Extract ORB features and find matches between two images."""
    orb = cv2.ORB_create(5000)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if descriptors1 is None or descriptors2 is None:
        return None, None, None
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return keypoints1, keypoints2, matches

def estimate_camera_pose(kp1, kp2, matches, camera_matrix):
    """Estimate the relative camera pose between two images."""
    if len(matches) < 8:
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None
    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, camera_matrix)
    return R, t

def create_point_cloud(rgb, depth, camera_matrix, pose=np.eye(4)):
    """Create an Open3D point cloud from RGB and depth images."""
    height, width = depth.shape
    fx, fy = camera_matrix[0,0], camera_matrix[1,1]
    cx, cy = camera_matrix[0,2], camera_matrix[1,2]

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x3 = (x - cx) * depth / fx
    y3 = (y - cy) * depth / fy
    z3 = depth

    points = np.stack((x3, y3, z3), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0

    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed = (pose @ points_hom.T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_transformed)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.remove_non_finite_points()
    return pcd

def reconstruct_map(scene_rgb_dir, depth_dir, output_map_file, camera_intrinsics):
    """Reconstructs the 3D map from RGB and depth images and saves it."""
    rgb_images, image_files = load_rgb_images(scene_rgb_dir)
    if not rgb_images:
        print(f"No RGB images found in {scene_rgb_dir}.")
        return

    depth_maps = load_depth_maps(depth_dir, image_files)
    # Modified line to fix the ValueError
    if not any(dm is not None for dm in depth_maps):
        print(f"No valid depth maps found in {depth_dir}.")
        return

    camera_matrix = np.array([
        [camera_intrinsics['fx'], 0, camera_intrinsics['cx']],
        [0, camera_intrinsics['fy'], camera_intrinsics['cy']],
        [0, 0, 1]
    ])

    current_pose = np.eye(4)
    point_clouds = []

    for i in tqdm(range(len(rgb_images)), desc="Processing Images"):
        rgb = rgb_images[i]
        depth = depth_maps[i]
        if depth is None:
            print(f"Skipping image {i} due to missing depth map.")
            continue

        pcd = create_point_cloud(rgb, depth, camera_matrix, current_pose)
        point_clouds.append(pcd)

        if i > 0:
            img1 = rgb_images[i - 1]
            img2 = rgb_images[i]

            kp1, kp2, matches = extract_features_and_matches(img1, img2)
            if matches is None or len(matches) < 8:
                print(f"Not enough matches between image {i - 1} and {i}. Skipping pose estimation.")
                continue

            R, t = estimate_camera_pose(kp1, kp2, matches, camera_matrix)
            if R is None or t is None:
                print(f"Pose estimation failed between image {i - 1} and {i}. Skipping.")
                continue

            transformation = np.eye(4)
            transformation[:3, :3] = R
            transformation[:3, 3] = t.ravel()
            current_pose = current_pose @ np.linalg.inv(transformation)

    combined_pcd = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        combined_pcd += pcd
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.02)

    os.makedirs(os.path.dirname(output_map_file), exist_ok=True)
    o3d.io.write_point_cloud(output_map_file, combined_pcd)
    print(f"3D map saved to {output_map_file}")

def visualize_map(output_map_file):
    """Visualizes the saved 3D map."""
    if not os.path.exists(output_map_file):
        print(f"Map file {output_map_file} does not exist. Cannot visualize.")
        return

    pcd = o3d.io.read_point_cloud(output_map_file)
    o3d.visualization.draw_geometries([pcd])

def main():
    scene_rgb_dir = 'data/rgbd_dataset_freiburg1_room/rgbd_dataset_freiburg1_room/rgb_subset'  # Path to RGB images
    depth_dir = 'data/rgbd_dataset_freiburg1_room/rgbd_dataset_freiburg1_room/depth'  # Path to depth maps
    output_map_file = 'results/freiburg1_room_map.ply'  # Path to save the 3D map

    camera_intrinsics = {
        'fx': 517.3,
        'fy': 516.5,
        'cx': 318.6,
        'cy': 255.3
    }

    reconstruct_map(scene_rgb_dir, depth_dir, output_map_file, camera_intrinsics)
    visualize_map(output_map_file)

if __name__ == "__main__":
    main()
