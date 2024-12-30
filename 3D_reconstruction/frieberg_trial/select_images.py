import cv2
import os
import glob
import shutil
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def calculate_histogram(image_path):
    """Calculate a color histogram for the image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV for better color separation
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def calculate_ssim(image1, image2):
    """Calculate Structural Similarity Index (SSIM) between two images."""
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

# Define paths
image_paths = sorted(glob.glob("data\\rgbd_dataset_freiburg1_room\\rgbd_dataset_freiburg1_room\\rgb\\*"))  # Ensure sorted order
target_directory = "data\\rgbd_dataset_freiburg1_room\\rgbd_dataset_freiburg1_room\\selected_images\\"
os.makedirs(target_directory, exist_ok=True)

# Set the maximum number of images to select
max_images = 100  # Adjust this value as needed
chunk_size = max(1, len(image_paths) // max_images)  # Calculate chunk size to evenly distribute selection

# Initialize variables
selected_images = []
prev_hist = None

# Process images by chunk
print(f"Selecting a maximum of {max_images} keyframes...")
for i in tqdm(range(0, len(image_paths), chunk_size), desc="Processing Chunks"):
    # Get the current chunk
    chunk = image_paths[i : i + chunk_size]
    best_image = None
    best_score = -1

    for image_path in chunk:
        current_hist = calculate_histogram(image_path)

        if prev_hist is not None:
            # Compare histograms for global change
            hist_diff = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)
            score = 1 - hist_diff  # Higher score for lower similarity
        else:
            score = 1  # First image always selected

        # Update best image in the chunk
        if score > best_score:
            best_score = score
            best_image = image_path

    if best_image:
        selected_images.append(best_image)
        prev_hist = calculate_histogram(best_image)  # Update previous histogram

    # Stop if we reach the maximum number of images
    if len(selected_images) >= max_images:
        break

# Copy selected images to the target directory
print("Copying selected images...")
for image_path in tqdm(selected_images, desc="Saving Images"):
    filename = os.path.basename(image_path)
    target_path = os.path.join(target_directory, filename)
    shutil.copy(image_path, target_path)

print(f"Selected {len(selected_images)} images out of {len(image_paths)}.")
print(f"Images have been saved to {target_directory}")
