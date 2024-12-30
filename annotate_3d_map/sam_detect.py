import os
import cv2
import json
import numpy as np
from ultralytics import SAM
from ultralytics.engine.results import Results
from tqdm import tqdm

def main():
    # Paths
    data_folder_path = "../data/scannet_data/scans/scene0000_00/rgbd"
    color_dir = os.path.join(data_folder_path, "color")

    input_dir = "output_yolo"
    output_dir = "output_sam"
    os.makedirs(output_dir, exist_ok=True)

    # Subfolders
    overlay_dir = os.path.join(output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    yolo_detections_path = os.path.join(input_dir, "yolo_detections.json")
    if not os.path.exists(yolo_detections_path):
        print(f"Error: {yolo_detections_path} does not exist. Run file1.py first.")
        return

    # Load YOLO detections
    with open(yolo_detections_path, 'r') as f:
        all_detections = json.load(f)

    # Initialize SAM model
    print("Loading MobileSAM model...")
    try:
        sam_model = SAM("mobile_sam.pt")
    except Exception as e:
        print(f"Error loading MobileSAM model: {e}")
        return

    # Convert keys to ints and sort
    image_indices = sorted([int(k) for k in all_detections.keys()])

    total_detections = 0
    processed_detections = 0

    for idx in tqdm(image_indices, desc="SAM Segmentation"):
        detections = all_detections.get(str(idx), [])
        if not detections:
            # No detections for this image
            continue

        image_path = os.path.join(color_dir, f"{idx}.jpg")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to load {image_path}. Skipping.")
            continue
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create an overlay image to visualize SAM masks for all detections
        overlay_sam = img.copy()

        for det_i, det in enumerate(detections):
            total_detections += 1
            label = det.get("class", "unknown")
            confidence = det.get("confidence", 0.0)
            bbox = det.get("bbox", [0, 0, 0, 0])

            if len(bbox) != 4:
                print(f"Warning: Invalid bbox format for detection {det_i} in image {idx}. Skipping.")
                continue

            x_min, y_min, x_max, y_max = bbox

            # Calculate the center of the bounding box
            u_center = int((x_min + x_max) / 2)
            v_center = int((y_min + y_max) / 2)

            # Define prompt points for SAM (using center point only)
            points = [[u_center, v_center]]
            labels_sam = [1]  # Positive prompt

            # Prepare directories for saving masks
            class_dir = os.path.join(output_dir, label.replace(" ", "_"))
            os.makedirs(class_dir, exist_ok=True)

            # Run SAM
            try:
                results = sam_model.predict(img_rgb, points=points, labels=labels_sam)
                if isinstance(results, list) and len(results) > 0:
                    first_result = results[0]
                    if isinstance(first_result, Results) and hasattr(first_result.masks, 'data'):
                        mask_array = first_result.masks.data.cpu().numpy()
                        if mask_array.ndim == 3:
                            mask = mask_array[0]
                        elif mask_array.ndim == 2:
                            mask = mask_array
                        else:
                            print(f"Warning: Unexpected mask shape for detection {det_i} in image {idx}. Skipping.")
                            continue

                        # Resize mask to full image dimensions if needed
                        if (mask.shape[1] != w) or (mask.shape[0] != h):
                            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

                        # Save mask as a binary image (255 for object, 0 for background)
                        mask_img = (mask.astype(np.uint8) * 255)
                        mask_filename = f"{idx}_{det_i}.png"
                        mask_path = os.path.join(class_dir, mask_filename)
                        cv2.imwrite(mask_path, mask_img)

                        # Update overlay_sam
                        # Create a colored mask overlay (e.g., semi-transparent green)
                        colored_mask = np.zeros_like(img, dtype=np.uint8)
                        colored_mask[mask] = (0, 255, 0)  # Green color for the mask

                        # Blend the colored mask with the overlay image
                        alpha = 0.3  # Transparency factor
                        overlay_sam = cv2.addWeighted(overlay_sam, 1.0, colored_mask, alpha, 0)

                        processed_detections += 1
                    else:
                        print(f"Warning: SAM did not return a valid Results object for detection {det_i} in image {idx}.")
                else:
                    print(f"Warning: SAM returned empty results for detection {det_i} in image {idx}.")
            except Exception as e:
                print(f"Error during SAM inference for detection '{label}' (ID: {det_i}) in image {idx}: {e}")
                continue

        # After processing all detections for this image, save the overlay
        if processed_detections > 0:
            overlay_path = os.path.join(overlay_dir, f"{idx}_overlay.jpg")
            cv2.imwrite(overlay_path, overlay_sam)

    print(f"SAM processing completed.")
    print(f"Total detections found: {total_detections}")
    print(f"Total detections processed: {processed_detections}")
    print(f"Masks and overlays saved in '{output_dir}' directory.")

if __name__ == "__main__":
    main()
