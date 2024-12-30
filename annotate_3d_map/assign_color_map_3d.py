import os
import json
import cv2
import gc
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import cKDTree
import random

def load_matrix(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist. Using identity matrix.")
        return np.eye(4)
    matrix = []
    with open(file_path, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            matrix.append(row)
    matrix = np.array(matrix)
    if matrix.shape != (4,4):
        print(f"Warning: {file_path} is not 4x4. Using identity matrix.")
        return np.eye(4)
    return matrix

def scale_intrinsics(intrinsic, depth_w, depth_h, color_w, color_h):
    sx = color_w / depth_w
    sy = color_h / depth_h
    scaled = intrinsic.copy()
    scaled[0,0]*=sx
    scaled[1,1]*=sy
    scaled[0,2]*=sx
    scaled[1,2]*=sy
    return scaled

def pixel_to_world(u, v, d_val, intrinsic, extrinsic_depth, pose, w, h, max_distance=100):
    if not (0 <= u < w and 0 <= v < h):
        return None
    if d_val <=0 or not np.isfinite(d_val):
        return None
    fx = intrinsic[0,0]
    fy = intrinsic[1,1]
    cx = intrinsic[0,2]
    cy = intrinsic[1,2]

    x_cam = (u - cx)*d_val/fx
    y_cam = (v - cy)*d_val/fy
    z_cam = d_val

    point_cam = np.array([x_cam,y_cam,z_cam,1.0])
    point_depth = extrinsic_depth @ point_cam
    point_world = pose @ point_depth
    p = point_world[:3]
    if np.any(np.isnan(p)) or np.linalg.norm(p)>max_distance:
        return None
    return p

def boxes_overlap(box1, box2):
    # box: [xmin, ymin, zmin, xmax, ymax, zmax]
    # Two boxes overlap if intervals overlap in all three dimensions
    return (box1[0] <= box2[3] and box1[3] >= box2[0]) and \
           (box1[1] <= box2[4] and box1[4] >= box2[1]) and \
           (box1[2] <= box2[5] and box1[5] >= box2[2])

def merge_objects(objects, new_points, cls_name, distance_threshold=0.5):
    """
    Attempt to merge new_points into existing objects of the same class.
    If no suitable object is found, create a new object entry.
    Objects are considered the same if:
      - They are the same class
      - Their centroids are close OR their bounding boxes overlap.
    """
    new_box = [
        float(np.min(new_points[:,0])),
        float(np.min(new_points[:,1])),
        float(np.min(new_points[:,2])),
        float(np.max(new_points[:,0])),
        float(np.max(new_points[:,1])),
        float(np.max(new_points[:,2]))
    ]
    new_centroid = new_points.mean(axis=0)

    merged = False
    for obj in objects:
        if obj['class'] == cls_name:
            centroid_obj = obj['points_sum']/obj['points_count']
            dist = np.linalg.norm(centroid_obj - new_centroid)
            # Check bounding box overlap
            box_obj = obj['bbox']
            overlap = boxes_overlap(box_obj, new_box)

            if dist <= distance_threshold or overlap:
                # Merge into this object
                obj['points_sum'] += new_points.sum(axis=0)
                obj['points_count'] += len(new_points)
                obj['bbox'][0] = min(obj['bbox'][0], new_box[0])
                obj['bbox'][1] = min(obj['bbox'][1], new_box[1])
                obj['bbox'][2] = min(obj['bbox'][2], new_box[2])
                obj['bbox'][3] = max(obj['bbox'][3], new_box[3])
                obj['bbox'][4] = max(obj['bbox'][4], new_box[4])
                obj['bbox'][5] = max(obj['bbox'][5], new_box[5])
                obj['merged_points'].append(new_points)
                merged = True
                break

    if not merged:
        # Create a new object
        return {
            'class': cls_name,
            'points_sum': new_points.sum(axis=0),
            'points_count': len(new_points),
            'bbox': new_box,
            'merged_points':[new_points]
        }
    else:
        return None

def remove_noise_from_object(all_points, nb_neighbors=20, std_ratio=2.0):
    """
    Remove statistical outliers from the object's point set to clean noise.
    """
    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(all_points)
    # Remove statistical outliers
    pcd_clean, ind = pcd_obj.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    clean_points = np.asarray(pcd_clean.points)
    return clean_points

def compute_density_center(points, radius=0.05):
    """
    Compute a density-based center by finding the point with the highest local density.
    Density is defined as the number of points within `radius` of each point.
    """
    tree = cKDTree(points)
    densities = [len(tree.query_ball_point(p, r=radius)) for p in points]
    max_idx = np.argmax(densities)
    return points[max_idx]

def main():
    print("Starting SAM-based coloring...")

    data_folder_path = "../data/scannet_data/scans/scene0000_00/rgbd"
    color_dir = os.path.join(data_folder_path, "color")
    depth_dir = os.path.join(data_folder_path, "depth")
    intrinsic_dir = os.path.join(data_folder_path, "intrinsic")
    pose_dir = os.path.join(data_folder_path, "pose")
    sam_input_dir = "output_sam"

    output_dir = "output_3d"
    os.makedirs(output_dir, exist_ok=True)

    # Load reconstructed map
    input_ply = "output_3d_map/output_model_points_full.ply"
    annotated_ply = os.path.join(output_dir,"output_model_points_annotated.ply")
    bleached_ply = os.path.join(output_dir,"output_model_points_bleached.ply")
    final_anno_path = os.path.join(output_dir,"final_3d_annotations.json")

    print(f"Loading reconstructed map from {input_ply}...")
    pcd = o3d.io.read_point_cloud(input_ply)
    if not pcd.has_points():
        print("Error: Point cloud has no points.")
        return

    map_points = np.asarray(pcd.points)
    if pcd.has_colors():
        vertex_colors_original = np.asarray(pcd.colors)
    else:
        vertex_colors_original = np.ones((map_points.shape[0],3))

    # For bleached version, start all white
    vertex_colors_bleached = np.ones((map_points.shape[0],3))
    # For annotated version on top of original colors
    vertex_colors_annotated = vertex_colors_original.copy()

    point_cloud_tree = cKDTree(map_points)

    # Classes and color map
    target_classes=[
        "chair","couch","bed","dining table","tv","laptop","mouse","remote",
        "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
        "book","clock","vase","potted plant","scissors","teddy bear","hair drier",
        "toothbrush","bowl","bottle","wine glass","cup","fork","knife","spoon"
    ]

    color_map={
        "chair": [1, 0, 0],
        "couch": [0, 1, 0],
        "bed": [0, 0, 1],
        "dining table": [1,1,0],
        "tv": [1,0,1],
        "laptop": [0,1,1],
        "mouse": [0.5,0,0],
        "remote": [0,0.5,0],
        "keyboard": [0,0,0.5],
        "cell phone": [0.5,0.5,0],
        "microwave": [0.5,0,0.5],
        "oven": [0,0.5,0.5],
        "toaster": [0.75,0.75,0],
        "sink": [0.75,0,0.75],
        "refrigerator": [0,0.75,0.75],
        "book": [0.5,0.25,0],
        "clock": [0.25,0.5,0],
        "vase": [0.25,0,0.5],
        "potted plant": [0,0.25,0],
        "scissors": [0.5,0,0],
        "teddy bear": [0.75,0.5,0],
        "hair drier": [0.25,0.25,0],
        "toothbrush": [0.25,0,0],
        "bowl": [0.5,0,0.25],
        "bottle": [0.1,0.1,0.9],
        "wine glass": [0.9,0.1,0.1],
        "cup": [0,0.25,0.5],
        "fork": [0.3,0.3,0.7],
        "knife": [0.7,0.3,0.3],
        "spoon": [0.3,0.7,0.3]
    }

    # Load intrinsics/extrinsics
    intrinsic_depth_path = os.path.join(intrinsic_dir, "intrinsic_depth.txt")
    extrinsic_depth_path = os.path.join(intrinsic_dir, "extrinsic_depth.txt")
    intrinsic_depth_orig = load_matrix(intrinsic_depth_path)
    extrinsic_depth = load_matrix(extrinsic_depth_path)
    if intrinsic_depth_orig.shape!=(4,4):
        intrinsic_depth_orig = np.eye(4)
    if extrinsic_depth.shape!=(4,4):
        extrinsic_depth = np.eye(4)

    merging_distance_threshold=0.5
    map_distance_threshold=0.02
    max_pixels_per_mask=2000
    min_detections = 2  # Keep at least 2 merges

    intrinsics_cache={}
    pose_cache={}

    obj_list=[]

    print("Processing SAM detections...")
    class_dirs = [d for d in os.listdir(sam_input_dir) if os.path.isdir(os.path.join(sam_input_dir,d)) and d!="overlays"]

    for class_dir in tqdm(class_dirs, desc="SAM Classes"):
        cls_name=class_dir.replace("_"," ")
        if cls_name not in target_classes:
            continue
        cpath=os.path.join(sam_input_dir,class_dir)
        mask_files=[f for f in os.listdir(cpath) if f.endswith('.png')]
        for mf in tqdm(mask_files, desc=f"Processing {cls_name}", leave=False):
            base_name=os.path.splitext(mf)[0]
            parts=base_name.split('_')
            if len(parts)!=2:
                continue
            try:
                image_idx=int(parts[0])
            except:
                continue

            mask_path=os.path.join(cpath,mf)
            mask_img=cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                continue
            mask_bool=(mask_img>0)
            if not np.any(mask_bool):
                continue

            depth_path=os.path.join(depth_dir,f"{image_idx}.png")
            color_path=os.path.join(color_dir,f"{image_idx}.jpg")
            pose_path=os.path.join(pose_dir,f"{image_idx}.txt")

            if not (os.path.exists(depth_path) and os.path.exists(color_path) and os.path.exists(pose_path)):
                continue

            depth_image=cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                continue
            depth=depth_image.astype(np.float32)/1000.0
            dh,dw=depth.shape

            color_image=cv2.imread(color_path,cv2.IMREAD_COLOR)
            if color_image is None:
                continue
            color_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
            ch,cw,_=color_image.shape

            # Resize depth if needed
            if (dh!=ch) or (dw!=cw):
                depth_resized=cv2.resize(depth_image,(cw,ch),interpolation=cv2.INTER_NEAREST).astype(np.float32)/1000.0
                scaled_intrinsic=scale_intrinsics(intrinsic_depth_orig,dw,dh,cw,ch)
            else:
                depth_resized=depth
                scaled_intrinsic=intrinsic_depth_orig.copy()

            if image_idx not in pose_cache:
                p=load_matrix(pose_path)
                if p.shape!=(4,4):
                    p=np.eye(4)
                pose_cache[image_idx]=p
            pose=pose_cache[image_idx]

            if image_idx not in intrinsics_cache:
                intrinsics_cache[image_idx]=scaled_intrinsic
            intrinsic_depth=intrinsics_cache[image_idx]

            v_coords,u_coords=np.where(mask_bool)
            if len(u_coords)>max_pixels_per_mask:
                sampled_indices=random.sample(range(len(u_coords)), max_pixels_per_mask)
                u_coords=u_coords[sampled_indices]
                v_coords=v_coords[sampled_indices]

            mask_points=[]
            for u_pix,v_pix in zip(u_coords,v_coords):
                d_val=depth_resized[v_pix,u_pix]
                p=pixel_to_world(u_pix,v_pix,d_val,intrinsic_depth,extrinsic_depth,pose,cw,ch)
                if p is not None:
                    mask_points.append(p)
            if len(mask_points)==0:
                continue
            mask_points=np.array(mask_points)

            new_obj=merge_objects(obj_list, mask_points, cls_name, distance_threshold=merging_distance_threshold)
            if new_obj is not None:
                obj_list.append(new_obj)

    # Filter objects: Only keep those with at least `min_detections` merges
    obj_list = [o for o in obj_list if len(o['merged_points']) >= min_detections]

    if len(obj_list)==0:
        print("No valid objects from SAM detections (or no objects with >=2 SAM detections).")
        # Just save map as is (both versions)
        # Normal annotated (no changes)
        pcd.colors=o3d.utility.Vector3dVector(vertex_colors_original)
        o3d.io.write_point_cloud(annotated_ply, pcd)

        # Bleached version (all white)
        bleached_colors = np.ones((len(pcd.points),3))
        pcd.colors=o3d.utility.Vector3dVector(bleached_colors)
        o3d.io.write_point_cloud(bleached_ply, pcd)

        print("Annotated cloud saved with no changes.")
        return

    final_objects=[]
    class_counts = {}
    for obj in obj_list:
        all_obj_points = np.vstack(obj['merged_points'])

        # Remove noise from the object's points
        clean_points = remove_noise_from_object(all_obj_points, nb_neighbors=20, std_ratio=2.0)
        if len(clean_points) == 0:
            continue

        # Recompute centroid and bounding box after noise removal
        centroid = clean_points.mean(axis=0).tolist()
        bbox = [
            float(np.min(clean_points[:,0])),
            float(np.min(clean_points[:,1])),
            float(np.min(clean_points[:,2])),
            float(np.max(clean_points[:,0])),
            float(np.max(clean_points[:,1])),
            float(np.max(clean_points[:,2]))
        ]

        # Compute density-based center
        density_center = compute_density_center(clean_points, radius=0.05).tolist()

        cls_name=obj['class']
        if cls_name not in class_counts:
            class_counts[cls_name]=1
        else:
            class_counts[cls_name]+=1
        obj_id = class_counts[cls_name]

        final_objects.append({
            'object_id':obj_id,
            'class':cls_name,
            'centroid':centroid,
            'bbox_3d':bbox,
            'density_center': density_center,
            'points': clean_points
        })

    if len(final_objects)==0:
        print("All objects removed after noise filtering.")
        # Just save map as is (both versions)
        pcd.colors=o3d.utility.Vector3dVector(vertex_colors_original)
        o3d.io.write_point_cloud(annotated_ply, pcd)
        pcd.colors=o3d.utility.Vector3dVector(np.ones((len(pcd.points),3)))
        o3d.io.write_point_cloud(bleached_ply, pcd)
        return

    print("Coloring map with objects...")

    for obj in tqdm(final_objects, desc="Coloring"):
        color=color_map.get(obj['class'],[1,1,1])
        pts=obj['points']
        distances, indices = cKDTree(map_points).query(pts,k=1)
        valid_matches=distances<=map_distance_threshold
        vertex_colors_annotated[indices[valid_matches],:]=color
        vertex_colors_bleached[indices[valid_matches],:]=color

    # Save annotated map with original background
    pcd.colors=o3d.utility.Vector3dVector(vertex_colors_annotated)
    o3d.io.write_point_cloud(annotated_ply, pcd)
    print(f"Annotated point cloud saved to {annotated_ply}")

    # Save annotated map with bleached background
    pcd.colors=o3d.utility.Vector3dVector(vertex_colors_bleached)
    o3d.io.write_point_cloud(bleached_ply, pcd)
    print(f"Bleached annotated point cloud saved to {bleached_ply}")

    annotations={'objects':[]}
    for obj in final_objects:
        annotations['objects'].append({
            'object_id': obj['object_id'],
            'class': obj['class'],
            'position': obj['centroid'],
            'bbox_3d': obj['bbox_3d'],
            'density_center': obj['density_center']
        })

    with open(final_anno_path,'w') as f:
        json.dump(annotations,f,indent=4)
    print(f"Final 3D annotations saved to {final_anno_path}")

    gc.collect()
    print("Done.")

if __name__=="__main__":
    main()
