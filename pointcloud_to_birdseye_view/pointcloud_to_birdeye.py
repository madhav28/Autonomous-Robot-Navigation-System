#!/usr/bin/env python3
"""
pointcloud_to_birdeye.py

This script converts an unoriented 3D point cloud (PLY file) of a house into a bird's eye view (2D floor plan).
It identifies the floor based on height, reduces noise in obstacles, and generates a 2D occupancy grid centered
based on the obstacles.

Dependencies:
    - open3d
    - numpy
    - matplotlib
    - scikit-learn

Usage:
    Adjust the parameters below and run the script:
        python pointcloud_to_birdeye.py
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import os
from sklearn.cluster import DBSCAN

def load_point_cloud(file_path):
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        print(f"Error: The point cloud '{file_path}' is empty or could not be read.")
        sys.exit(1)
    print(f"Loaded point cloud from '{file_path}' with {len(pcd.points)} points.")
    return pcd

def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    print("Removing statistical outliers...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_clean = pcd.select_by_index(ind)
    print(f"Point cloud cleaned: {len(pcd_clean.points)} points remaining.")
    return pcd_clean

def identify_floor(points, floor_threshold=0.2):
    print("Identifying floor points based on height...")
    min_z = np.min(points[:, 2])
    floor_mask = points[:, 2] <= (min_z + floor_threshold)
    floor_points = points[floor_mask]
    obstacles_points = points[~floor_mask]
    print(f"Floor points identified: {len(floor_points)}")
    print(f"Obstacle points before noise reduction: {len(obstacles_points)}")
    return floor_points, obstacles_points

def reduce_obstacle_noise(obstacle_points, eps=0.3, min_samples=30):
    print("Reducing noise on obstacles using DBSCAN clustering...")
    if len(obstacle_points) == 0:
        print("No obstacle points to process.")
        return obstacle_points
    # Perform DBSCAN clustering on 2D (X,Y) projections
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(obstacle_points[:, :2])
    labels = clustering.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label
    if not unique_labels:
        print("No clusters found. All obstacles considered as noise.")
        return np.empty((0, 3))
    # Keep points belonging to clusters (labels >= 0)
    clustered_obstacles = obstacle_points[labels >= 0]
    print(f"Obstacle points after noise reduction: {len(clustered_obstacles)}")
    return clustered_obstacles

def create_occupancy_grid(floor_points, obstacle_points, grid_size=10.0, resolution=0.05, padding=1.0):
    print("Creating occupancy grid...")
    
    # Combine all points to compute bounds
    all_points = np.vstack((floor_points, obstacle_points))
    min_x, min_y = np.min(all_points[:, :2], axis=0)
    max_x, max_y = np.max(all_points[:, :2], axis=0)
    spread_x = max_x - min_x
    spread_y = max_y - min_y

    # Adjust grid_size if necessary
    required_grid_size = max(spread_x, spread_y) + 2 * padding
    if required_grid_size > grid_size:
        print(f"Adjusting grid size from {grid_size}m to {required_grid_size:.2f}m to accommodate all points.")
        grid_size = required_grid_size
    else:
        print(f"Using grid size: {grid_size}m")

    grid_cells = int(grid_size / resolution)
    occupancy_grid = np.zeros((grid_cells, grid_cells), dtype=np.int8)  # 0: Outside, 1: Floor, 2: Obstacles

    # Compute centroid
    centroid_x = (max_x + min_x) / 2
    centroid_y = (max_y + min_y) / 2
    print(f"Point cloud centroid: ({centroid_x:.2f}, {centroid_y:.2f})")

    # Shift points to center the grid
    floor_x = floor_points[:, 0] - centroid_x
    floor_y = floor_points[:, 1] - centroid_y
    obstacle_x = obstacle_points[:, 0] - centroid_x
    obstacle_y = obstacle_points[:, 1] - centroid_y

    half_size = grid_size / 2

    # Function to convert coordinates to grid indices
    def to_grid(x, y):
        i = np.floor((x + half_size) / resolution).astype(int)
        j = np.floor((y + half_size) / resolution).astype(int)
        return i, j

    # Assign floor points
    floor_i, floor_j = to_grid(floor_x, floor_y)

    # Filter out points outside the grid
    valid_floor = (floor_i >= 0) & (floor_i < grid_cells) & (floor_j >= 0) & (floor_j < grid_cells)
    floor_i = floor_i[valid_floor]
    floor_j = floor_j[valid_floor]

    # Assign floor
    occupancy_grid[floor_i, floor_j] = 1  # 1 represents floor

    # Assign obstacle points
    obstacle_i, obstacle_j = to_grid(obstacle_x, obstacle_y)

    # Filter out points outside the grid
    valid_obstacle = (obstacle_i >= 0) & (obstacle_i < grid_cells) & (obstacle_j >= 0) & (obstacle_j < grid_cells)
    obstacle_i = obstacle_i[valid_obstacle]
    obstacle_j = obstacle_j[valid_obstacle]

    # Assign obstacles
    occupancy_grid[obstacle_i, obstacle_j] = 2  # 2 represents obstacles

    # Handle hollow areas by removing floor points inside obstacle boundaries
    print("Correcting hollow regions...")
    for oi, oj in zip(obstacle_i, obstacle_j):
        if occupancy_grid[oi, oj] == 2:  # Check if the cell is an obstacle
            occupancy_grid[oi, oj] = 2  # Ensure obstacle cells override any potential floor cells

    print(f"Occupancy grid created with grid size: {grid_size} meters and resolution: {resolution} meters/pixel.")
    return occupancy_grid, centroid_x, centroid_y, grid_size


def plot_occupancy_grid(occupancy_grid, centroid_x, centroid_y, grid_size=10.0, resolution=0.05, output_path='birdeye_view.png'):
    print("Plotting occupancy grid...")
    cmap = colors.ListedColormap(['blue', 'red'])  # Only two colors now: Floor (blue) and Obstacles (red)
    bounds = [1, 2, 3]  # Corresponds to 1: Floor, 2: Obstacles
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 10))
    extent = [
        -grid_size / 2,
         grid_size / 2,
        -grid_size / 2,
         grid_size / 2
    ]
    plt.imshow(occupancy_grid.T, cmap=cmap, norm=norm, origin='lower', extent=extent)
    plt.title("Occupancy Grid Bird's Eye View")
    plt.xlabel("X-axis (meters)")
    plt.ylabel("Y-axis (meters)")
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Floor'),
        Patch(facecolor='red', edgecolor='black', label='Obstacles')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(output_path)
    plt.close()
    print(f"Occupancy grid saved to '{output_path}'.")

def main():
    # =========================
    # === User-Defined Parameters ===
    # =========================
    input_ply_path = "output_model_mesh_full.ply"    # <-- Set your PLY file path here
    output_image_path = "birdeye_view.png"           # Output image path
    grid_size = 10.0                                 # Initial size of the occupancy grid in meters
    resolution = 0.05                                # Resolution in meters per pixel
    padding = 1.0                                    # Padding around the point cloud in meters
    floor_threshold = 0.2                            # Threshold to classify floor points above min_z
    dbscan_eps = 0.3                                 # DBSCAN epsilon parameter for obstacle clustering
    dbscan_min_samples = 30                          # DBSCAN min_samples parameter for obstacle clustering

    # =========================
    # === Processing Steps ===
    # =========================

    # 1. Load Point Cloud
    pcd = load_point_cloud(input_ply_path)

    # 2. Remove Statistical Outliers
    pcd = remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0)

    # 3. Convert Point Cloud to NumPy Array
    points = np.asarray(pcd.points)

    # 4. Identify Floor and Obstacles Based on Height
    floor_points, obstacle_points = identify_floor(points, floor_threshold=floor_threshold)

    # 5. Reduce Noise on Obstacles using DBSCAN
    clustered_obstacles = reduce_obstacle_noise(obstacle_points, eps=dbscan_eps, min_samples=dbscan_min_samples)

    # 6. Create Occupancy Grid
    occupancy_grid, centroid_x, centroid_y, final_grid_size = create_occupancy_grid(
        floor_points, 
        clustered_obstacles,
        grid_size=grid_size, 
        resolution=resolution, 
        padding=padding
    )

    # 7. Plot and Save Occupancy Grid
    plot_occupancy_grid(
        occupancy_grid, 
        centroid_x, 
        centroid_y, 
        grid_size=final_grid_size, 
        resolution=resolution, 
        output_path=output_image_path
    )

if __name__ == "__main__":
    main()
