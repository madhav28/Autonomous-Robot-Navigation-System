import os
import open3d as o3d
import sys
import gc
import numpy as np
import argparse

def load_point_cloud(ply_path):
    """
    Load a point cloud from a PLY file.

    Parameters:
    - ply_path: str, path to the PLY file.

    Returns:
    - pcd: open3d.geometry.PointCloud, the loaded point cloud.
    """
    if not os.path.exists(ply_path):
        print(f"Error: The file {ply_path} does not exist.")
        sys.exit(1)
    
    print(f"Loading point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        print("Error: Loaded point cloud is empty.")
        sys.exit(1)
    print(f"Point cloud successfully loaded: {len(pcd.points)} points.")
    return pcd

def preprocess_point_cloud(pcd, voxel_size=0.02):
    """
    Preprocess the point cloud: downsample and estimate normals.

    Parameters:
    - pcd: open3d.geometry.PointCloud, the input point cloud.
    - voxel_size: float, the size of the voxel for downsampling.

    Returns:
    - pcd_down: open3d.geometry.PointCloud, the downsampled point cloud.
    """
    print(f"Downsampling the point cloud with voxel size = {voxel_size}...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"Downsampled point cloud has {len(pcd_down.points)} points.")

    print("Estimating normals...")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    pcd_down.orient_normals_consistent_tangent_plane(100)
    return pcd_down

def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Remove statistical outliers from the point cloud.

    Parameters:
    - pcd: open3d.geometry.PointCloud, the input point cloud.
    - nb_neighbors: int, number of neighbors to analyze for each point.
    - std_ratio: float, standard deviation ratio.

    Returns:
    - pcd_clean: open3d.geometry.PointCloud, the cleaned point cloud.
    """
    print("Removing statistical outliers...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_clean = pcd.select_by_index(ind)
    print(f"Point cloud after outlier removal has {len(pcd_clean.points)} points.")
    return pcd_clean

def generate_mesh_poisson(pcd, depth=9, scale=1.1, linear_fit=True):
    """
    Generate a mesh using Poisson surface reconstruction.

    Parameters:
    - pcd: open3d.geometry.PointCloud, the input point cloud with normals.
    - depth: int, the depth parameter for Poisson reconstruction.
    - scale: float, the scale parameter for bounding box.
    - linear_fit: bool, whether to use linear fitting.

    Returns:
    - mesh: open3d.geometry.TriangleMesh, the reconstructed mesh.
    """
    print("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit
    )
    print(f"Poisson reconstruction completed: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")

    # Remove low density vertices to clean the mesh
    print("Removing low density vertices...")
    densities = np.asarray(densities)
    density_threshold = np.percentile(densities, 5)  # Adjust percentile as needed
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"Mesh after density-based cleaning: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")

    return mesh

def generate_mesh_ball_pivoting(pcd, radii=[0.005, 0.01, 0.02, 0.04]):
    """
    Generate a mesh using the Ball Pivoting algorithm.

    Parameters:
    - pcd: open3d.geometry.PointCloud, the input point cloud with normals.
    - radii: list of float, radii to use for ball pivoting.

    Returns:
    - mesh: open3d.geometry.TriangleMesh, the reconstructed mesh.
    """
    print("Running Ball Pivoting algorithm...")
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        print(f"Ball Pivoting completed: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")
        return mesh
    except Exception as e:
        print(f"Ball Pivoting failed: {e}")
        return None

def clean_mesh(mesh, mesh_cleaning=True):
    """
    Clean the mesh by removing non-manifold edges and small holes.

    Parameters:
    - mesh: open3d.geometry.TriangleMesh, the input mesh.
    - mesh_cleaning: bool, whether to perform mesh cleaning.

    Returns:
    - mesh_clean: open3d.geometry.TriangleMesh, the cleaned mesh.
    """
    if not mesh_cleaning:
        return mesh

    print("Cleaning mesh...")
    # Remove duplicated triangles and vertices
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    print(f"Mesh after cleaning: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")
    return mesh

def smooth_mesh(mesh, number_of_iterations=10, lambda_smooth=0.5, lambda_constrain=1.0):
    """
    Apply Laplacian smoothing to the mesh.

    Parameters:
    - mesh: open3d.geometry.TriangleMesh, the input mesh.
    - number_of_iterations: int, number of smoothing iterations.
    - lambda_smooth: float, smoothing factor.
    - lambda_constrain: float, constraining factor.

    Returns:
    - mesh_smoothed: open3d.geometry.TriangleMesh, the smoothed mesh.
    """
    print("Smoothing mesh...")
    mesh_smoothed = mesh.filter_smooth_simple(number_of_iterations=number_of_iterations)
    print("Mesh smoothing completed.")
    return mesh_smoothed

def transfer_vertex_colors(pcd, mesh):
    """
    Transfer vertex colors from the point cloud to the mesh.

    Parameters:
    - pcd: open3d.geometry.PointCloud, the input point cloud.
    - mesh: open3d.geometry.TriangleMesh, the input mesh.

    Returns:
    - mesh: open3d.geometry.TriangleMesh, the mesh with vertex colors.
    """
    print("Transferring vertex colors to the mesh...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    colors = []
    for vertex in mesh.vertices:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
        if k > 0:
            colors.append(pcd.colors[idx[0]])
        else:
            colors.append([0, 0, 0])  # Default color
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    print("Vertex colors transferred.")
    return mesh

def save_mesh(mesh, output_path):
    """
    Save the mesh to a file.

    Parameters:
    - mesh: open3d.geometry.TriangleMesh, the mesh to save.
    - output_path: str, path to save the mesh file.
    """
    print(f"Saving mesh to {output_path}...")
    success = o3d.io.write_triangle_mesh(output_path, mesh)
    if success:
        print("Mesh successfully saved.")
    else:
        print("Error: Failed to save the mesh.")

def visualize(pcd, mesh):
    """
    Visualize the point cloud and the mesh.

    Parameters:
    - pcd: open3d.geometry.PointCloud, the input point cloud.
    - mesh: open3d.geometry.TriangleMesh, the reconstructed mesh.
    """
    print("Visualizing the point cloud and mesh...")
    pcd_temp = pcd.paint_uniform_color([0.5, 0.5, 0.5])
    mesh_temp = mesh.paint_uniform_color([1.0, 0.75, 0.0])
    o3d.visualization.draw_geometries([pcd_temp, mesh_temp],
                                      zoom=0.8,
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    - args: argparse.Namespace, the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Enhanced 3D Mesh Generation from Point Cloud")
    parser.add_argument('--input_ply', type=str, default="output_3d_map/output_model_points_full.ply",
                        help="Path to the input PLY point cloud file.")
    parser.add_argument('--output_dir', type=str, default="output_mesh",
                        help="Directory to save the generated mesh.")
    parser.add_argument('--meshing_method', type=str, choices=['poisson', 'ball_pivoting'], default='poisson',
                        help="Meshing algorithm to use.")
    parser.add_argument('--depth', type=int, default=9,
                        help="Depth parameter for Poisson reconstruction.")
    parser.add_argument('--scale', type=float, default=1.1,
                        help="Scale parameter for Poisson reconstruction.")
    parser.add_argument('--linear_fit', type=bool, default=True,
                        help="Linear fit parameter for Poisson reconstruction.")
    parser.add_argument('--voxel_size', type=float, default=0.02,
                        help="Voxel size for downsampling.")
    parser.add_argument('--smoothing_iterations', type=int, default=10,
                        help="Number of iterations for mesh smoothing.")
    parser.add_argument('--no_visualize', action='store_true',
                        help="Do not visualize the mesh after generation.")
    parser.add_argument('--remove_outliers', action='store_true',
                        help="Remove statistical outliers from the point cloud before meshing.")
    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = parse_arguments()

    input_ply = args.input_ply
    output_dir = args.output_dir
    meshing_method = args.meshing_method
    depth = args.depth
    scale = args.scale
    linear_fit = args.linear_fit
    voxel_size = args.voxel_size
    smoothing_iterations = args.smoothing_iterations
    no_visualize = args.no_visualize
    remove_outliers = args.remove_outliers

    os.makedirs(output_dir, exist_ok=True)

    # Load point cloud
    pcd = load_point_cloud(input_ply)

    # Preprocess point cloud
    pcd_down = preprocess_point_cloud(pcd, voxel_size=voxel_size)

    if remove_outliers:
        pcd_down = remove_statistical_outliers(pcd_down)

    # Choose meshing method
    if meshing_method == 'poisson':
        # Generate mesh using Poisson reconstruction
        mesh = generate_mesh_poisson(pcd_down, depth=depth, scale=scale, linear_fit=linear_fit)
    elif meshing_method == 'ball_pivoting':
        # Generate mesh using Ball Pivoting
        mesh = generate_mesh_ball_pivoting(pcd_down)
        if mesh is None:
            print("Ball Pivoting failed. Exiting.")
            sys.exit(1)
    else:
        print(f"Unknown meshing method: {meshing_method}. Exiting.")
        sys.exit(1)

    # Clean the mesh
    mesh = clean_mesh(mesh)

    # Smooth the mesh
    mesh = smooth_mesh(mesh, number_of_iterations=smoothing_iterations)

    # Optional: Transfer colors from point cloud to mesh
    if pcd_down.has_colors():
        mesh = transfer_vertex_colors(pcd_down, mesh)

    # Save mesh
    output_mesh = os.path.join(output_dir, f"reconstructed_mesh_{meshing_method}.ply")
    save_mesh(mesh, output_mesh)

    # Cleanup
    pcd, pcd_down, mesh = None, None, None
    gc.collect()
    print("Mesh generation completed.")

if __name__ == "__main__":
    main()
