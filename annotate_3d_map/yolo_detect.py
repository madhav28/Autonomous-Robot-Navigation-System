import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def main():
    # Paths
    data_folder_path = "../data/scannet_data/scans/scene0000_00/rgbd"
    color_dir = os.path.join(data_folder_path, "color")

    output_dir = "output_yolo"
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv8 model (e.g. yolov8n)
    print("Loading YOLOv8 model...")
    try:
        yolo_model = YOLO("yolov8l.pt")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Target classes (same as before)
    target_classes = [
    "chair", "couch", "bed", "dining table",
    "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", 
    "sink", "refrigerator", "book", "clock", "vase", 
    "potted plant", "scissors", "teddy bear", 
    "hair drier", "toothbrush", "bowl", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon"
    ]

    # Verify classes are in YOLO model's dataset
    yolo_coco_classes = yolo_model.names
    print(yolo_coco_classes.values())
    missing_classes = [cls for cls in target_classes if cls not in yolo_coco_classes.values()]
    if missing_classes:
        print(f"Warning: The following classes are not in YOLOv8 COCO classes: {missing_classes}")
        target_classes = [cls for cls in target_classes if cls not in missing_classes]

    # Get list of color images
    image_files = [f for f in os.listdir(color_dir) if f.endswith('.jpg')]
    image_indices = sorted([int(os.path.splitext(f)[0]) for f in image_files])
    # Process only every 25th image
    image_indices = image_indices[::10]

    print(f"Running YOLO on {len(image_indices)} images...")

    all_detections = {}  # {img_index: [{"class": str, "confidence": float, "bbox": [x_min, y_min, x_max, y_max]}]}

    for idx in tqdm(image_indices, desc="YOLO Inference"):
        image_path = os.path.join(color_dir, f"{idx}.jpg")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to load {image_path}. Skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = yolo_model(img_rgb)

        detections = []
        for result in results:
            for box in result.boxes:
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                label = yolo_model.names[cls_id]
                if label not in target_classes or conf < 0.5:
                    continue
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                detections.append({
                    "class": label,
                    "confidence": float(conf),
                    "bbox": [x_min, y_min, x_max, y_max]
                })

        # Draw bounding boxes on the image
        overlay = img.copy()
        for det in detections:
            x_min, y_min, x_max, y_max = det["bbox"]
            cv2.rectangle(overlay, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 2)
            cv2.putText(overlay, f"{det['class']}: {det['confidence']:.2f}", (int(x_min), int(y_min)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Save the overlay image
        overlay_path = os.path.join(output_dir, f"{idx}_yolo_overlay.jpg")
        cv2.imwrite(overlay_path, overlay)

        all_detections[idx] = detections

    # Save detections to JSON
    detections_json_path = os.path.join(output_dir, "yolo_detections.json")
    with open(detections_json_path, 'w') as f:
        json.dump(all_detections, f, indent=4)
    print(f"Detections saved to {detections_json_path}")


if __name__ == "__main__":
    main()
