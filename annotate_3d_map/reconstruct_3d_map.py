import os
import cv2
import gc
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import cKDTree

def load_matrix(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist. Using identity matrix.")
        return np.eye(4)
    matrix = []
    with open(file_path, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            matrix.append(row)
    matrix = np.array(matrix)
    if matrix.shape != (4,4):
        print(f"Warning: {file_path} is not 4x4. Using identity matrix.")
        return np.eye(4)
    return matrix

def scale_intrinsics(intrinsic, depth_w, depth_h, color_w, color_h):
    sx = color_w / depth_w
    sy = color_h / depth_h
    scaled = intrinsic.copy()
    scaled[0,0]*=sx
    scaled[1,1]*=sy
    scaled[0,2]*=sx
    scaled[1,2]*=sy
    return scaled

def main():
    print("Starting 3D Reconstruction...")

    data_folder_path = "../data/scannet_data/scans/scene0000_00/rgbd"
    color_dir = os.path.join(data_folder_path, "color")
    depth_dir = os.path.join(data_folder_path, "depth")
    intrinsic_dir = os.path.join(data_folder_path, "intrinsic")
    pose_dir = os.path.join(data_folder_path, "pose")

    output_dir = "output_3d_map"
    os.makedirs(output_dir, exist_ok=True)

    output_ply = os.path.join(output_dir, "output_model_points_full.ply")

    # Load intrinsics/extrinsics
    intrinsic_depth_path = os.path.join(intrinsic_dir, "intrinsic_depth.txt")
    extrinsic_depth_path = os.path.join(intrinsic_dir, "extrinsic_depth.txt")
    intrinsic_depth_orig = load_matrix(intrinsic_depth_path)
    extrinsic_depth = load_matrix(extrinsic_depth_path)
    if intrinsic_depth_orig.shape!=(4,4):
        intrinsic_depth_orig = np.eye(4)
    if extrinsic_depth.shape!=(4,4):
        extrinsic_depth = np.eye(4)

    # Gather image indices
    image_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
    image_indices = sorted([int(os.path.splitext(f)[0]) for f in image_files])
    image_indices = image_indices[::25]  # use every 25th image
    print(f"Reconstructing 3D map from {len(image_indices)} images...")

    all_points=[]
    all_colors=[]

    for idx in tqdm(image_indices, desc="Building map"):
        depth_path = os.path.join(depth_dir, f"{idx}.png")
        color_path = os.path.join(color_dir, f"{idx}.jpg")
        pose_path = os.path.join(pose_dir, f"{idx}.txt")

        if not (os.path.exists(depth_path) and os.path.exists(color_path) and os.path.exists(pose_path)):
            continue

        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            continue
        depth = depth_image.astype(np.float32)/1000.0
        dh, dw = depth.shape

        color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
        if color_image is None:
            continue
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        ch, cw, _ = color_image.shape

        # Resize depth to color resolution if needed
        if (dh!=ch) or (dw!=cw):
            depth_resized = cv2.resize(depth_image,(cw,ch),interpolation=cv2.INTER_NEAREST).astype(np.float32)/1000.0
            scaled_intrinsic = scale_intrinsics(intrinsic_depth_orig, dw, dh, cw, ch)
        else:
            depth_resized = depth
            scaled_intrinsic = intrinsic_depth_orig.copy()

        pose = load_matrix(pose_path)
        if pose.shape!=(4,4):
            pose=np.eye(4)

        uu, vv = np.meshgrid(np.arange(cw), np.arange(ch))
        uu=uu.flatten()
        vv=vv.flatten()
        depths=depth_resized.flatten()
        valid=(depths>0)&np.isfinite(depths)
        uu=uu[valid]
        vv=vv[valid]
        depths=depths[valid]

        fx=scaled_intrinsic[0,0]
        fy=scaled_intrinsic[1,1]
        cx=scaled_intrinsic[0,2]
        cy=scaled_intrinsic[1,2]

        x_cam=(uu - cx)*depths/fx
        y_cam=(vv - cy)*depths/fy
        z_cam=depths

        points_cam = np.vstack((x_cam,y_cam,z_cam,np.ones_like(z_cam))).T
        points_depth = (extrinsic_depth @ points_cam.T).T
        points_world = (pose @ points_depth.T).T[:,:3]

        colors = color_image[vv,uu,:]/255.0

        all_points.append(points_world)
        all_colors.append(colors)

    if len(all_points)==0:
        print("No valid points, cannot reconstruct map.")
        return

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(all_points)
    pcd.colors=o3d.utility.Vector3dVector(all_colors)
    all_points,all_colors=None,None
    gc.collect()

    # Downsample map
    voxel_size=0.02
    pcd=pcd.voxel_down_sample(voxel_size)
    print(f"Map after downsampling: {len(pcd.points)} points.")

    # Save the reconstructed map
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"Reconstructed 3D map saved to {output_ply}")

    gc.collect()
    print("3D Reconstruction done.")

if __name__=="__main__":
    main()
