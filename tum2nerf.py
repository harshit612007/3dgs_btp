import os, json, math
import numpy as np
from scipy.spatial.transform import Rotation

def read_tum_txt(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if not parts: continue
            data.append(parts)
    return data

def main():
    base_dir = "datasets/desk2"
    
    # 1. Read RGB and Groundtruth
    rgb_data = read_tum_txt(os.path.join(base_dir, "rgb.txt"))
    gt_data = read_tum_txt(os.path.join(base_dir, "groundtruth.txt"))
    
    # Parse GT: timestamp -> (tx, ty, tz, qx, qy, qz, qw)
    gt_dict = {}
    for row in gt_data:
        gt_dict[float(row[0])] = np.array([float(x) for x in row[1:]])
    
    gt_times = np.array(sorted(list(gt_dict.keys())))
    
    # 2. Match timestamps
    frames = []
    for row in rgb_data:
        rgb_time = float(row[0])
        rgb_path = row[1]
        
        # Find closest GT time
        idx = np.searchsorted(gt_times, rgb_time)
        if idx == 0: closest_time = gt_times[0]
        elif idx == len(gt_times): closest_time = gt_times[-1]
        else:
            t1, t2 = gt_times[idx-1], gt_times[idx]
            closest_time = t1 if abs(rgb_time - t1) < abs(rgb_time - t2) else t2
            
        # Must be within 20ms
        if abs(rgb_time - closest_time) > 0.02:
            continue
            
        pose = gt_dict[closest_time]
        tx, ty, tz, qx, qy, qz, qw = pose
        
        # TUM quaternion is qx, qy, qz, qw
        # Convert to 4x4 matrix
        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = rot
        c2w[:3, 3] = [tx, ty, tz]
        
        # Convert TUM (OpenCV) coordinates to NeRF (OpenGL) coordinates!
        # OpenCV: right, down, forward
        # OpenGL/NeRF: right, up, back
        # We must flip Y and Z axes!
        c2w[:, 1:3] *= -1
        
        frames.append({
            "file_path": rgb_path,
            "transform_matrix": c2w.tolist()
        })
        
    # TUM camera intrinsics (Freiburg1)
    # fx = 517.3, fy = 516.5, cx = 318.6, cy = 255.3, w = 640, h = 480
    # NeRF json expects camera_angle_x (FOV in radians)
    camera_angle_x = 2.0 * math.atan(640 / (2.0 * 517.3))
    
    out_dict = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }
    
    with open(os.path.join(base_dir, "transforms_train.json"), "w") as f:
        json.dump(out_dict, f, indent=4)
        
    print(f"Generated transforms_train.json with {len(frames)} valid matched frames!")

if __name__ == '__main__':
    main()
