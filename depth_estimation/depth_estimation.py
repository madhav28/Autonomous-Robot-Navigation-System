# depth_estimation.py

import os

def main():
    model_type = "dpt_swin2_tiny_256"  # Choose the desired MiDaS model type
    # input_path = "data/rgbd_dataset_freiburg1_room/rgbd_dataset_freiburg1_room/rgb"  # Path to RGB images
    # output_path = "data/rgbd_dataset_freiburg1_room/rgbd_dataset_freiburg1_room/depth"  # Path to save depth maps
    input_path = "data/scannet_data/color"
    output_path = "data/scannet_data/depth"

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Construct the command to run MiDaS
    command = f"python .\\MiDaS\\run.py --model_type {model_type} --input_path {input_path} --output_path {output_path}"

    # Execute the command
    os.system(command)

if __name__ == "__main__":
    main()
