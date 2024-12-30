import os
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm

def load_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as f:
        for line in f:
            row = [float(num) for num in line.strip().split()]
            matrix.append(row)
    return np.array(matrix)

def backproject_depth(depth, intrinsic, downsample_factor=2):
    if downsample_factor > 1:
        depth = depth[::downsample_factor, ::downsample_factor]
        intrinsic = intrinsic.copy()
        intrinsic[0, 0] /= downsample_factor
        intrinsic[1, 1] /= downsample_factor
        intrinsic[0, 2] /= downsample_factor
        intrinsic[1, 2] /= downsample_factor

    height, width = depth.shape
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()
    depth = depth.flatten()

    valid = depth > 0
    uu = uu[valid]
    vv = vv[valid]
    depth = depth[valid]

    x = (uu - cx) * depth / fx
    y = (vv - cy) * depth / fy
    z = depth
    points = np.vstack((x, y, z)).T
    return points, uu, vv

def main():
    data_folder_path = "../data/scannet_data/scans/scene0000_00/rgbd"
    color_dir = os.path.join(data_folder_path, "color")
    depth_dir = os.path.join(data_folder_path, "depth")
    intrinsic_dir = os.path.join(data_folder_path, "intrinsic")
    pose_dir = os.path.join(data_folder_path, "pose")

    output_ply = "outputs/output_model_points.ply"
    output_mesh_ply = "outputs/output_model_mesh.ply"

    intrinsic_depth_path = os.path.join(intrinsic_dir, "intrinsic_depth.txt")
    extrinsic_depth_path = os.path.join(intrinsic_dir, "extrinsic_depth.txt")
    intrinsic_color_path = os.path.join(intrinsic_dir, "intrinsic_color.txt")
    extrinsic_color_path = os.path.join(intrinsic_dir, "extrinsic_color.txt")

    intrinsic_depth = load_matrix(intrinsic_depth_path)
    extrinsic_depth = load_matrix(extrinsic_depth_path)
    intrinsic_color = load_matrix(intrinsic_color_path)
    extrinsic_color = load_matrix(extrinsic_color_path)

    if extrinsic_depth.size == 0:
        extrinsic_depth = np.eye(4)
    if extrinsic_color.size == 0:
        extrinsic_color = np.eye(4)

    image_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
    image_indices = sorted([int(os.path.splitext(f)[0]) for f in image_files])

    # Limit the number of images to process
    # image_indices = image_indices[:num_images_to_process]
    image_indices = image_indices[::20]

    pcd = o3d.geometry.PointCloud()
    print(f"Starting 3D reconstruction on {len(image_indices)} images...")

    # Process images
    for idx in tqdm(image_indices, desc="Processing images"):
        try:
            depth_path = os.path.join(depth_dir, f"{idx}.png")
            color_path = os.path.join(color_dir, f"{idx}.jpg")
            pose_path = os.path.join(pose_dir, f"{idx}.txt")

            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                continue
            depth = depth_image.astype(np.float32) / 1000.0

            color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
            if color_image is None:
                continue
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            pose = load_matrix(pose_path)

            points_camera, uu, vv = backproject_depth(depth, intrinsic_depth, downsample_factor=2)

            points_camera_hom = np.hstack((points_camera, np.ones((points_camera.shape[0], 1))))
            points_depth = (extrinsic_depth @ points_camera_hom.T).T[:, :3]

            points_world_hom = np.hstack((points_depth, np.ones((points_depth.shape[0], 1))))
            points_world = (pose @ points_world_hom.T).T[:, :3]

            h_color, w_color, _ = color_image.shape
            valid_indices = (uu >= 0) & (uu < w_color) & (vv >= 0) & (vv < h_color)
            points_world = points_world[valid_indices]
            uu_valid = uu[valid_indices].astype(int)
            vv_valid = vv[valid_indices].astype(int)
            colors = color_image[vv_valid, uu_valid, :] / 255.0

            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(points_world)
            temp_pcd.colors = o3d.utility.Vector3dVector(colors)

            pcd += temp_pcd

            if len(pcd.points) > 2000000:
                pcd = pcd.voxel_down_sample(voxel_size=0.02)

        except Exception:
            continue

    if len(pcd.points) == 0:
        print("No valid points processed.")
        return

    # Remove outliers to improve quality
    pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]

    # Final downsampling
    voxel_size = 0.02
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Points after final downsampling: {len(pcd.points)}")

    o3d.io.write_point_cloud(output_ply, pcd)

    # Estimate normals with better parameters
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=50))
    pcd.orient_normals_consistent_tangent_plane(30)

    # Poisson reconstruction with slightly deeper depth for more detail
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    densities = np.asarray(densities)
    density_threshold = np.percentile(densities, 5)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Optional smoothing to improve mesh quality
    mesh = mesh.filter_smooth_taubin(number_of_iterations=5)

    # Simplify while preserving quality
    target_number_of_triangles = 500000
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    o3d.io.write_triangle_mesh(output_mesh_ply, mesh)
    print(f"Mesh saved to {output_mesh_ply}")

if __name__ == "__main__":
    main()


#for v2 results:
#target_number_of_triangles = 500000
#image_indices = image_indices[::50]
#voxel_size = 0.03