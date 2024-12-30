import os
import json
import cv2
import numpy as np
import open3d as o3d
import yaml  # Added for YAML file handling

def main():
    print("Generating bird's-eye view...")

    output_dir = "output_birdeye"
    os.makedirs(output_dir, exist_ok=True)

    # Load point cloud map
    map_ply = "output_3d_map/output_model_points_full.ply"
    if not os.path.exists(map_ply):
        print("Error: Map file not found. Run reconstruction first.")
        return

    pcd = o3d.io.read_point_cloud(map_ply)
    if not pcd.has_points():
        print("Error: No points in map.")
        return
    points = np.asarray(pcd.points)

    # Determine XY bounding box of the scene
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # Scale: 100 pixels per meter
    scale = 100
    width = int((max_x - min_x) * scale) + 1
    height = int((max_y - min_y) * scale) + 1
    if width <= 0 or height <= 0:
        print("Invalid map dimensions.")
        return

    # Create visualization image (white background)
    bird_img_vis = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Create binary map image
    # Binary image: Red for obstacles, Blue for floor
    floor_color_bin = (255, 0, 0)      # Blue in BGR
    obstacle_color_bin = (0, 0, 255)   # Red in BGR
    bird_img_bin = np.ones((height, width, 3), dtype=np.uint8) * np.array(floor_color_bin, dtype=np.uint8)

    # Visualization colors (brighter and more distinct):
    # Floor: pure blue, Obstacles: pure red
    floor_color_vis = (255, 0, 0)    # Blue in BGR
    obstacle_color_vis = (0, 0, 255) # Red in BGR

    floor_threshold = 0.1

    # Fill images with floor/obstacle colors based on Z
    for p in points:
        x, y, z = p
        u = int((x - min_x) * scale)
        v = int((max_y - y) * scale)  # invert y
        if 0 <= u < width and 0 <= v < height:
            if z < floor_threshold:
                # Floor
                bird_img_vis[v, u] = floor_color_vis
                bird_img_bin[v, u] = floor_color_bin
            else:
                # Obstacle
                bird_img_vis[v, u] = obstacle_color_vis
                bird_img_bin[v, u] = obstacle_color_bin

    # Load final 3D annotations
    anno_path = "output_3d/final_3d_annotations.json"
    if not os.path.exists(anno_path):
        print("No annotations found. Saving maps without object annotations.")
        cv2.imwrite(os.path.join(output_dir, "birdseye_view_visual.png"), bird_img_vis)
        cv2.imwrite(os.path.join(output_dir, "birdseye_view_binary.png"), bird_img_bin)
        print(f"Birdseye views saved in '{output_dir}' as 'birdseye_view_visual.png' and 'birdseye_view_binary.png'")
        return

    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    # Prepare to collect 2D locations for YAML output
    object_locations = []

    # Plot objects using density centers
    for obj in annotations['objects']:
        cls = obj['class']
        obj_id = obj['object_id']
        label = f"{cls}{obj_id}"
        density_center = obj.get('density_center', None)

        if density_center is None or len(density_center) != 3:
            # Skip if no valid density_center
            continue

        dc_x, dc_y, dc_z = density_center
        center_x = int((dc_x - min_x) * scale)
        center_y = int((max_y - dc_y) * scale)

        if 0 <= center_x < width and 0 <= center_y < height:
            # Draw a circle at the density center
            cv2.circle(bird_img_vis, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)  # Filled green circle

            # Draw label near the center
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness_text = 1
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness_text)
            text_w, text_h = text_size

            # Position text slightly below the center
            text_x, text_y = center_x - text_w // 2, center_y + text_h + 6

            # Adjust if text goes out of boundaries
            if text_x < 0:
                text_x = 0
            elif text_x + text_w > bird_img_vis.shape[1]:
                text_x = bird_img_vis.shape[1] - text_w

            if text_y + 4 > bird_img_vis.shape[0]:
                text_y = center_y - 6

            # Background rectangle for text
            cv2.rectangle(bird_img_vis, (text_x, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), (0, 255, 0), -1)

            # Black text on green background
            cv2.putText(bird_img_vis, label, (text_x + 1, text_y), font, font_scale, (0, 0, 0), thickness_text, cv2.LINE_AA)

            # Collect 2D location for YAML
            object_locations.append({
                'object_id': obj_id,
                'class': cls,
                'label': label,
                'location': [center_x, center_y]
            })

    # Remove random noise from obstacles in the binary image
    # Create a mask for obstacles (red pixels)
    obstacle_mask = np.all(bird_img_bin == (0, 0, 255), axis=-1).astype(np.uint8) * 255

    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    obstacle_mask_clean = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)

    # Reconstruct the binary image with cleaned obstacle mask
    cleaned_bin_img = np.zeros_like(bird_img_bin)
    cleaned_bin_img[:] = floor_color_bin
    cleaned_bin_img[obstacle_mask_clean == 255] = obstacle_color_bin

    # Save final images
    visual_path = os.path.join(output_dir, "birdseye_view_visual.png")
    binary_path = os.path.join(output_dir, "birdseye_view_binary.png")

    cv2.imwrite(visual_path, bird_img_vis)
    cv2.imwrite(binary_path, cleaned_bin_img)

    print(f"Birdseye views saved in '{output_dir}' as 'birdseye_view_visual.png' and 'birdseye_view_binary.png'.")

    # Save 2D locations to YAML file
    yaml_path = os.path.join(output_dir, "object_locations.yaml")
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump({'objects': object_locations}, yaml_file, default_flow_style=False)

    print(f"2D object locations saved to '{yaml_path}'.")

if __name__ == "__main__":
    main()
