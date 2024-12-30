# generate_annotations.py

import os
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
from ultralytics import YOLO, SAM
import json
from collections import defaultdict
import gc
from ultralytics.engine.results import Results

def load_matrix(file_path):
    """
    Load a 4x4 matrix from a text file.

    Parameters:
        file_path (str): Path to the matrix text file.

    Returns:
        np.ndarray: 4x4 numpy array.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Matrix file {file_path} does not exist. Using identity matrix.")
        return np.eye(4)
        
    matrix = []
    with open(file_path, 'r') as f:
        for line in f:
            row = [float(num) for num in line.strip().split()]
            matrix.append(row)
    return np.array(matrix)

def pixel_to_world(u, v, depth, intrinsic, pose, image_width, image_height):
    """
    Convert pixel coordinates to world coordinates with boundary checks.

    Parameters:
        u (float): Pixel x-coordinate.
        v (float): Pixel y-coordinate.
        depth (float): Depth value in meters.
        intrinsic (np.ndarray): 4x4 intrinsic matrix.
        pose (np.ndarray): 4x4 pose matrix.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        np.ndarray or None: 3D point in world coordinates or None if out of bounds.
    """
    # Boundary checks
    if not (0 <= u < image_width) or not (0 <= v < image_height):
        return None

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    point_camera = np.array([x, y, z, 1.0])  # Homogeneous coordinates
    point_world = pose @ point_camera
    return point_world[:3]

def is_valid_point(point):
    """
    Check if a 3D point contains valid (non-NaN, non-Infinity) values.

    Parameters:
        point (np.ndarray): 3D point.

    Returns:
        bool: True if valid, False otherwise.
    """
    return np.all(np.isfinite(point))

def is_reasonable_position(point, max_distance=100):
    """
    Check if the 3D point is within a reasonable distance.

    Parameters:
        point (np.ndarray): 3D point.
        max_distance (float): Maximum allowable distance from the origin.

    Returns:
        bool: True if within distance, False otherwise.
    """
    return np.linalg.norm(point) <= max_distance

def detect_objects(image_path, yolo_model, conf_threshold=0.25):
    """
    Detect objects in an image using YOLOv8.

    Parameters:
        image_path (str): Path to the RGB image.
        yolo_model (YOLO): YOLOv8 model instance.
        conf_threshold (float): Confidence threshold for detections.

    Returns:
        list of dict: Detected objects with class, confidence, and bounding box.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to load image {image_path}")
        return []

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = yolo_model(img_rgb)

    detections = []
    for result in results:
        boxes = result.boxes  # Boxes object for each image
        for box in boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            if conf < conf_threshold:
                continue
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            label = yolo_model.names[cls]
            detections.append({
                'class': label,
                'confidence': conf,
                'bbox': [x_min, y_min, x_max, y_max]
            })

    return detections

def get_sam_mask(image_path, points, labels, sam_model, output_size):
    """
    Get the segmentation mask from MobileSAM based on point prompts.

    Parameters:
        image_path (str): Path to the RGB image.
        points (list of list): List of points for prompts.
        labels (list): Corresponding labels for the points.
        sam_model (SAM): MobileSAM model instance.
        output_size (tuple): Desired output size as (width, height).

    Returns:
        np.ndarray or None: Binary mask of the segmentation, resized to the output size.
    """
    try:
        # Load image as RGB
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to load image {image_path} for SAM.")
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform prediction using SAM
        results = sam_model.predict(img_rgb, points=points, labels=labels)

        # Debugging: Print type and content structure of results
        print(f"SAM prediction results type: {type(results)}")
        if isinstance(results, list) and len(results) > 0:
            first_result = results[0]
            print(f"SAM prediction first_result type: {type(first_result)}")

            if isinstance(first_result, Results):
                print(f"Results masks type: {type(first_result.masks)}")

                # Extract mask data from the Results object
                masks = first_result.masks
                if hasattr(masks, 'data'):
                    mask_array = masks.data.cpu().numpy()  # Convert to NumPy array
                    if mask_array.ndim == 3:
                        mask = mask_array[0]  # Get the first mask
                    elif mask_array.ndim == 2:
                        mask = mask_array  # Single mask scenario
                    else:
                        print(f"Warning: Unexpected mask dimensions {mask_array.shape}")
                        return None

                    # Resize mask to match the output size (depth image size)
                    desired_width, desired_height = output_size  # Correct ordering
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8),
                        (desired_width, desired_height),  # (width, height)
                        interpolation=cv2.INTER_NEAREST
                    )

                    # Ensure that mask_resized has the exact desired dimensions
                    mask_resized = mask_resized[:desired_height, :desired_width]

                    # Check the dimensions
                    print(f"Resized mask shape: {mask_resized.shape}, Desired shape: ({desired_height}, {desired_width})")

                    return mask_resized.astype(bool)

                else:
                    print(f"Warning: Masks in Results object have no 'data' attribute.")
                    return None
            else:
                print(f"Warning: First result is not a Results object.")
                return None
        else:
            print(f"Warning: SAM returned unexpected result format for {image_path}.")
            return None

    except Exception as e:
        print(f"Error during SAM prediction for {image_path}: {e}")
        return None

def extract_3d_points_from_mask(mask, depth_image, pose, intrinsic, image_width, image_height):
    """
    Extract 3D points from the segmentation mask.

    Parameters:
        mask (np.ndarray): Binary segmentation mask.
        depth_image (np.ndarray): Depth image in meters.
        pose (np.ndarray): 4x4 pose matrix.
        intrinsic (np.ndarray): 4x4 intrinsic matrix.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        np.ndarray: Array of 3D points.
    """
    # Find pixel indices where mask is True
    indices = np.where(mask)
    v_coords = indices[0]
    u_coords = indices[1]

    # Debugging: Check maximum indices
    if v_coords.size == 0 or u_coords.size == 0:
        print("Warning: No valid mask pixels found.")
        return np.array([])

    print(f"Mask indices max v: {v_coords.max()}, max u: {u_coords.max()}")
    print(f"Depth image shape: {depth_image.shape}")

    # Ensure indices are within depth image bounds
    if v_coords.max() >= depth_image.shape[0] or u_coords.max() >= depth_image.shape[1]:
        print(f"Error: Mask indices exceed depth image dimensions.")
        return np.array([])

    # Extract depth values for these pixels
    depths = depth_image[v_coords, u_coords]

    # Filter out invalid depths
    valid = (depths > 0) & np.isfinite(depths)
    u_valid = u_coords[valid]
    v_valid = v_coords[valid]
    depths_valid = depths[valid]

    # Convert to world coordinates
    points_world = []
    for u, v, d in zip(u_valid, v_valid, depths_valid):
        point = pixel_to_world(u, v, d, intrinsic, pose, image_width, image_height)
        if point is not None and is_valid_point(point) and is_reasonable_position(point):
            points_world.append(point)

    if not points_world:
        print("Warning: No valid 3D points extracted from mask.")
        return np.array([])

    return np.array(points_world)

def downsample_points(points, voxel_size=0.05):
    """
    Downsample 3D points using a voxel grid filter.

    Parameters:
        points (np.ndarray): Nx3 array of points.
        voxel_size (float): Size of the voxel grid in meters.

    Returns:
        np.ndarray: Downsampled Nx3 array of points.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd_down.points)

def apply_nms(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to filter overlapping detections.

    Parameters:
        detections (list of dict): List of detections with 'bbox' and 'confidence'.
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        list of dict: Detections after NMS.
    """
    if not detections:
        return []

    boxes = np.array([det['bbox'] for det in detections])
    confidences = np.array([det['confidence'] for det in detections])
    classes = np.array([det['class'] for det in detections])

    # Convert boxes to [x1, y1, x2, y2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confidences.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [detections[idx] for idx in keep]

def main():
    # Paths (modify these paths if your directory structure is different)
    data_folder_path = "scannet_data/scans/scene0000_00/rgbd"
    color_dir = os.path.join(data_folder_path, "color")
    depth_dir = os.path.join(data_folder_path, "depth")
    intrinsic_dir = os.path.join(data_folder_path, "intrinsic")
    pose_dir = os.path.join(data_folder_path, "pose")

    # Output file
    annotations_file = "annotations.json"

    # Load intrinsic and extrinsic matrices
    intrinsic_depth_path = os.path.join(intrinsic_dir, "intrinsic_depth.txt")
    extrinsic_depth_path = os.path.join(intrinsic_dir, "extrinsic_depth.txt")
    intrinsic_color_path = os.path.join(intrinsic_dir, "intrinsic_color.txt")
    extrinsic_color_path = os.path.join(intrinsic_dir, "extrinsic_color.txt")

    intrinsic_depth = load_matrix(intrinsic_depth_path)
    extrinsic_depth = load_matrix(extrinsic_depth_path)
    intrinsic_color = load_matrix(intrinsic_color_path)
    extrinsic_color = load_matrix(extrinsic_color_path)

    # If intrinsic or extrinsic matrices are not provided, assume identity
    if intrinsic_depth.size == 0:
        intrinsic_depth = np.eye(4)
    if extrinsic_depth.size == 0:
        extrinsic_depth = np.eye(4)
    if intrinsic_color.size == 0:
        intrinsic_color = np.eye(4)
    if extrinsic_color.size == 0:
        extrinsic_color = np.eye(4)

    # Initialize YOLOv8 model (YOLOv8n)
    print("Loading YOLOv8 model...")
    try:
        yolo_model = YOLO("yolov8n.pt")  # Ensure 'yolov8n.pt' is downloaded or provide the correct path
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return

    # Initialize MobileSAM model
    print("Loading MobileSAM model...")
    try:
        sam_model = SAM("mobile_sam.pt")  # Ensure 'mobile_sam.pt' is downloaded or provide the correct path
    except Exception as e:
        print(f"Error loading MobileSAM model: {e}")
        return

    # Define the list of classes to include (as per the user's specification)
    target_classes = [
        # Furniture and Fixtures
        "chair", "couch", "bed", "dining table", "toilet",
        # Appliances and Electronics
        "tv", "laptop", "mouse", "remote", "keyboard",
        "microwave", "oven", "toaster", "sink", "refrigerator",
        # Household Objects
        "book", "clock", "vase", "scissors", "teddy bear",
        "hair dryer", "toothbrush",
        # Other Indoor Items
        "potted plant", "cup", "bowl"
    ]

    # Verify that all target classes are in the YOLOv8 COCO dataset
    yolo_coco_classes = yolo_model.names  # Dictionary mapping class indices to names
    missing_classes = [cls for cls in target_classes if cls not in yolo_coco_classes.values()]
    if missing_classes:
        print(f"Warning: The following classes are not present in YOLOv8 COCO classes and will be ignored: {missing_classes}")
        # Remove missing classes from target_classes
        target_classes = [cls for cls in target_classes if cls not in missing_classes]

    # Define approximate dimensions for each class in meters (width, height, depth)
    size_map = {
        "chair": [0.5, 0.9, 0.5],
        "couch": [1.5, 0.8, 0.7],
        "bed": [2.0, 0.6, 1.5],
        "dining table": [1.5, 0.75, 1.0],
        "toilet": [0.5, 0.7, 0.5],
        "tv": [1.0, 0.6, 0.1],
        "laptop": [0.3, 0.02, 0.2],
        "mouse": [0.1, 0.05, 0.02],
        "remote": [0.15, 0.05, 0.02],
        "keyboard": [0.5, 0.02, 0.2],
        "microwave": [0.5, 0.4, 0.4],
        "oven": [0.6, 0.6, 0.6],
        "toaster": [0.3, 0.2, 0.3],
        "sink": [0.6, 0.4, 0.5],
        "refrigerator": [0.7, 1.8, 0.7],
        "book": [0.2, 0.03, 0.3],
        "clock": [0.3, 0.3, 0.05],
        "vase": [0.2, 0.4, 0.2],
        "scissors": [0.15, 0.05, 0.05],
        "teddy bear": [0.3, 0.3, 0.3],
        "hair dryer": [0.2, 0.3, 0.2],
        "toothbrush": [0.05, 0.02, 0.02],
        "potted plant": [0.4, 0.6, 0.4],
        "cup": [0.1, 0.1, 0.1],
        "bowl": [0.2, 0.1, 0.2]
    }

    # Initialize a list to store unique objects
    unique_objects = []

    # Define the distance threshold for assigning detections to unique objects (in meters)
    distance_threshold = 0.5  # Adjust based on your data scale

    # Get list of image indices (assuming filenames are sequential numbers)
    image_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
    image_indices = sorted([int(os.path.splitext(f)[0]) for f in image_files])

    # Limit the number of images to process (for testing purposes)
    image_indices = image_indices[::10]  # Change this value as needed

    print(f"Starting annotation on {len(image_indices)} images...")

    # Create directories to save failed masks and empty point masks (optional)
    failed_masks_dir = "failed_masks"
    empty_masks_dir = "empty_masks"
    os.makedirs(failed_masks_dir, exist_ok=True)
    os.makedirs(empty_masks_dir, exist_ok=True)

    # Initialize counters for logging
    total_detections = 0
    skipped_detections = 0

    for idx in tqdm(image_indices, desc="Processing images"):
        try:
            # Paths for current image
            depth_path = os.path.join(depth_dir, f"{idx}.png")
            color_path = os.path.join(color_dir, f"{idx}.jpg")
            pose_path = os.path.join(pose_dir, f"{idx}.txt")

            # Load depth image
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                print(f"Warning: Unable to load depth image {depth_path}. Skipping.")
                continue
            depth = depth_image.astype(np.float32) / 1000.0  # Convert to meters

            # Load color image
            color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
            if color_image is None:
                print(f"Warning: Unable to load color image {color_path}. Skipping.")
                continue
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Load pose
            pose = load_matrix(pose_path)

            # Get image dimensions
            depth_height, depth_width = depth_image.shape
            image_height, image_width = depth_height, depth_width  # Assuming color and depth images have same dimensions

            # Detect objects in the color image using YOLOv8
            detections = detect_objects(color_path, yolo_model, conf_threshold=0.25)
            if not detections:
                print(f"No detections found in image {idx}.")
                continue

            # Apply Non-Maximum Suppression (NMS) to reduce overlapping detections
            detections = apply_nms(detections, iou_threshold=0.5)
            if not detections:
                print(f"No detections found in image {idx} after NMS.")
                continue

            for det in detections:
                total_detections += 1
                label = det['class']
                if label not in target_classes:
                    print(f"Skipping detection with label '{label}' as it's not in target_classes.")
                    skipped_detections += 1
                    continue  # Ignore classes not in the target list
                bbox = det['bbox']
                confidence = det['confidence']

                # Define point prompts for SAM based on the bounding box
                x_min, y_min, x_max, y_max = bbox
                u_center = int((x_min + x_max) / 2)
                v_center = int((y_min + y_max) / 2)

                # Define multiple points within the bounding box for better segmentation
                # Here, we'll define 5 points: center and four corners
                points = [
                    [int(x_min), int(y_min)],
                    [int(x_max), int(y_min)],
                    [int(x_min), int(y_max)],
                    [int(x_max), int(y_max)],
                    [u_center, v_center]
                ]
                labels_sam = [1] * len(points)  # All points are positive prompts

                # Ensure points are within image boundaries
                points_clipped = []
                for point in points:
                    u, v = point
                    u = max(0, min(u, image_width - 1))
                    v = max(0, min(v, image_height - 1))
                    points_clipped.append([u, v])

                # Set output size as (width, height) for cv2.resize
                output_size = (image_width, image_height)

                # Get SAM mask
                mask = get_sam_mask(color_path, points_clipped, labels_sam, sam_model, output_size=output_size)
                if mask is None:
                    print(f"Skipping detection '{label}' due to SAM mask generation failure.")
                    # Optionally save the failed mask
                    # cv2.imwrite(f"{failed_masks_dir}/mask_{idx}_{label}.png", mask.astype(np.uint8)*255)
                    skipped_detections += 1
                    continue

                # Extract 3D points from the mask
                points_world = extract_3d_points_from_mask(mask, depth, pose, intrinsic_depth, image_width, image_height)
                if points_world.size == 0:
                    print(f"Skipping detection '{label}' as no valid 3D points were extracted from the mask.")
                    # Optionally save the empty mask
                    cv2.imwrite(f"{empty_masks_dir}/mask_{idx}_{label}.png", mask.astype(np.uint8)*255)
                    skipped_detections += 1
                    continue

                # Optional: Downsample points to reduce memory usage
                points_world = downsample_points(points_world, voxel_size=0.05)
                if points_world.size == 0:
                    print(f"Skipping detection '{label}' as no valid 3D points were extracted from the mask after downsampling.")
                    skipped_detections += 1
                    continue

                # Compute the centroid of the detected points
                centroid = np.mean(points_world, axis=0)

                # Assign the detection to a unique object based on proximity
                assigned = False
                for obj in unique_objects:
                    # Calculate the current centroid of the unique object
                    unique_centroid = obj['position_sum'] / obj['points_count']
                    distance = np.linalg.norm(unique_centroid - centroid)
                    if distance <= distance_threshold:
                        # Assign to this unique object
                        obj['points_count'] += len(points_world)
                        obj['confidence_sum'] += confidence * len(points_world)
                        obj['position_sum'] += points_world.sum(axis=0)  # Element-wise addition
                        # Update bounding box
                        obj['bbox_3d'][0] = min(obj['bbox_3d'][0], float(np.min(points_world[:, 0])))
                        obj['bbox_3d'][1] = min(obj['bbox_3d'][1], float(np.min(points_world[:, 1])))
                        obj['bbox_3d'][2] = min(obj['bbox_3d'][2], float(np.min(points_world[:, 2])))
                        obj['bbox_3d'][3] = max(obj['bbox_3d'][3], float(np.max(points_world[:, 0])))
                        obj['bbox_3d'][4] = max(obj['bbox_3d'][4], float(np.max(points_world[:, 1])))
                        obj['bbox_3d'][5] = max(obj['bbox_3d'][5], float(np.max(points_world[:, 2])))
                        # Update class counts
                        obj['class_counts'][label] += 1
                        print(f"Assigned detection '{label}' to existing unique object.")
                        assigned = True
                        break

                if not assigned:
                    # Create a new unique object
                    new_object = {
                        'class_counts': defaultdict(int),
                        'confidence_sum': confidence * len(points_world),
                        'points_count': len(points_world),
                        'position_sum': points_world.sum(axis=0),  # Initialize as numpy array
                        'bbox_3d': [
                            float(np.min(points_world[:, 0])),
                            float(np.min(points_world[:, 1])),
                            float(np.min(points_world[:, 2])),
                            float(np.max(points_world[:, 0])),
                            float(np.max(points_world[:, 1])),
                            float(np.max(points_world[:, 2]))
                        ]
                    }
                    new_object['class_counts'][label] += 1
                    unique_objects.append(new_object)
                    print(f"Created a new unique object for detection '{label}'.")

        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue

    if not unique_objects:
        print("No objects were detected. Exiting.")
        exit()

    print(f"Total unique objects detected: {len(unique_objects)}")
    print(f"Total detections processed: {total_detections}")
    print(f"Total detections skipped: {skipped_detections}")

    # Aggregate annotations based on unique_objects
    annotations = {'objects': []}
    for obj in unique_objects:
        # Determine the majority class
        classes = list(obj['class_counts'].keys())
        counts = list(obj['class_counts'].values())
        majority_class = classes[np.argmax(counts)]

        # Calculate average confidence
        avg_confidence = obj['confidence_sum'] / obj['points_count']

        # Calculate average position
        avg_position = (obj['position_sum'] / obj['points_count']).tolist()

        # Bounding box
        bbox_3d = obj['bbox_3d']

        annotations['objects'].append({
            'class': majority_class,
            'confidence': avg_confidence,
            'position': avg_position,
            'bbox_3d': bbox_3d
            # 'points': obj['points'].tolist()  # Omit saving individual points to reduce size
        })

    # Save only aggregated annotations to JSON
    combined_annotations = {
        'aggregated_annotations': annotations
    }

    try:
        with open(annotations_file, 'w') as f:
            json.dump(combined_annotations, f, indent=4)
        print(f"Annotations saved to {annotations_file}")
    except Exception as e:
        print(f"Error saving annotations: {e}")

    # Clean up memory
    gc.collect()

if __name__ == "__main__":
    main()
