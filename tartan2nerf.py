import os, json, math
import numpy as np
from scipy.spatial.transform import Rotation

def main():
    # You will need to change this path to point exactly to the 'easy' trajectory folder inside TartanAir
    base_dir = "tartanair_data/ArchVizTinyHouseDay/Easy/P000" 
    
    pose_file = os.path.join(base_dir, "pose_left.txt")
    img_dir = os.path.join(base_dir, "image_left")
    
    if not os.path.exists(pose_file):
        print(f"Error: Could not find {pose_file}. Please check the path.")
        return
        
    # 1. Read TartanAir Poses (tx ty tz qx qy qz qw)
    with open(pose_file, 'r') as f:
        poses = [[float(x) for x in line.strip().split()] for line in f if line.strip()]
        
    # Get all image paths (they map 1-to-1 with the lines in pose_left.txt)
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    if len(poses) != len(img_files):
        print(f"Warning: Number of poses ({len(poses)}) does not match number of images ({len(img_files)})!")
    
    frames = []
    for i in range(min(len(poses), len(img_files))):
        tx, ty, tz, qx, qy, qz, qw = poses[i]
        
        # Convert Quaternions to 3x3 Rotation Matrix
        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = rot
        c2w[:3, 3] = [tx, ty, tz]
        
        # TartanAir uses OpenCV coordinates (right, down, forward)
        # We must flip Y and Z to match NeRF/OpenGL (right, up, back)
        c2w[:, 1:3] *= -1
        
        # We only store the relative path from the dataset root folder
        rel_img_path = os.path.join("image_left", img_files[i]).replace("\\", "/")
        
        frames.append({
            "file_path": rel_img_path,
            "transform_matrix": c2w.tolist()
        })
        
    # 2. Normalize Poses (Crucial for TartanAir!)
    # TartanAir environments are massive. We must scale them so the cameras fit in a unit sphere.
    translations = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])
    center = np.mean(translations, axis=0)
    max_dist = np.max(np.linalg.norm(translations - center, axis=1))
    
    for f in frames:
        c2w = np.array(f["transform_matrix"])
        c2w[:3, 3] -= center # Center the path at origin
        c2w[:3, 3] /= (max_dist * 1.5) # Scale to fit comfortably inside the unit sphere
        f["transform_matrix"] = c2w.tolist()
        
    # TartanAir Camera Intrinsics
    # fx = 320.0, fy = 320.0, cx = 320.0, cy = 240.0, w = 640, h = 480
    camera_angle_x = 2.0 * math.atan(640 / (2.0 * 320.0))
    
    out_dict = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }
    
    out_json = os.path.join(base_dir, "transforms_train.json")
    with open(out_json, "w") as f:
        json.dump(out_dict, f, indent=4)
        
    print(f"Success! Generated {out_json} with {len(frames)} valid frames!")
    print(f"You can now run: python train.py --dataset_path {base_dir} --output_dir outputs/tartan_test --iterations 30000 --num_points 50000")

if __name__ == '__main__':
    main()
