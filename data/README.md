# Instructions to Download and Process the ScanNet Dataset

## About the Dataset
The **ScanNet dataset** is a comprehensive RGB-D video dataset featuring:
- Over 2.5 million views from more than 1,500 indoor scene scans.
- Annotated 3D camera poses, surface reconstructions, and instance-level semantic segmentations.

### Dataset Components
The dataset provides:
1. **RGB-D Sequences**: Color and depth video sequences captured by 3D sensors.
2. **Camera Intrinsics and Extrinsics**: Calibration information.
3. **3D Reconstructions**: Densely reconstructed 3D models.
4. **Semantic Labels**: Ground truth annotations for surfaces and images.
5. **Instance Labels**: Segmentation data at the object level.

### Focus for This Project
We will specifically use the data for **`scene0000`** in this project.

---

## Step 1: Request Access and Obtain Dataset
1. Visit the [official ScanNet website](http://www.scan-net.org/).
2. Sign the Terms of Use Agreement to request access.
3. After approval, you’ll receive instructions for downloading the dataset and scripts to help retrieve specific scenes.

---

## Step 2: Download the Dataset
Use the provided script `download-scannet.py` to download the required scene (`scene0000_00`).

Run the following command:
```bash
python download-scannet.py -o ./scannet_dataset --id scene0000_00
```

This command will:
- Save the downloaded files in the `./scannet_dataset` directory.
- Specifically download data for `scene0000_00`.

---

## Step 3: Extract RGBD Images and Camera Poses from `.sens` Data
The downloaded `.sens` file needs to be processed to extract RGBD images, depth images, camera intrinsics, and poses.

### Clone the ScanNet Repository
Download the ScanNet repository to use its tools:
```bash
git clone https://github.com/ScanNet/ScanNet.git
```

Navigate to the Python tools for `.sens` data:
```bash
cd ScanNet/SensReader/python/
```

### Replace the `SensorData.py` File
The provided `SensorData.py` in this directory must be updated to ensure compatibility with Python 3. Replace it with the modified `SensorData.py` file available in the **map_reconstruction** directory of this repo. Fix any minor Python 3 compatibility issues as necessary.

---

## Step 4: Prepare the Output Directory
Create a folder named `rgbd` inside the `scene0000_00` directory:

```bash
mkdir ../../../../data/scannet_dataset/scans/scene0000_00/rgbd"
```

---

## Step 5: Run the Data Reader Script
Use the `reader.py` script to extract the data:
```bash
python reader.py \
--filename ../../../../data/scannet_dataset/scans/scene0000_00/scene0000_00.sens \
--output_path ../../../../data/scannet_dataset/scans/scene0000_00/rgbd \
--export_color_images \
--export_depth_images \
--export_intrinsics \
--export_poses
```

This command will:
- Extract **RGB images**, **depth images**, **camera intrinsics**, and **camera poses**.
- Save them in the `rgbd` folder under `scene0000_00`.

---

## Final Directory Structure
After processing, the directory structure for `scene0000_00` will look like this:
```
scannet_dataset/
└── scans/
    └── scene0000_00/
        ├── scene0000_00.sens     # Original .sens file
        └── rgbd/
            ├── color/            # RGB images
            ├── depth/            # Depth images
            ├── intrinsics.txt    # Camera intrinsics
            └── poses/            # Camera poses
```

You now have the RGB images, depth images, and other metadata extracted and ready for use.

--- 