import os
import sys
import json
import argparse
import numpy as np
import math

# Add Scaffold-GS to path so we can import its colmap loader
sys.path.append(os.path.join(os.path.dirname(__file__), "Scaffold-GS"))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, qvec2rotmat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to COLMAP sparse/0 directory")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    cameras_extrinsic_file = os.path.join(args.source, "images.bin")
    cameras_intrinsic_file = os.path.join(args.source, "cameras.bin")
    
    if not os.path.exists(cameras_extrinsic_file) or not os.path.exists(cameras_intrinsic_file):
        print(f"COLMAP binary files not found in {args.source}")
        return

    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    
    # Process points3D
    try:
        xyzs, rgbs, errors = read_points3D_binary(os.path.join(args.source, "points3D.bin"))
        from plyfile import PlyData, PlyElement
        
        # Save points3d.ply
        vertex = np.empty(len(xyzs), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                           ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertex['x'] = xyzs[:, 0]
        vertex['y'] = xyzs[:, 1]
        vertex['z'] = xyzs[:, 2]
        vertex['red'] = rgbs[:, 0]
        vertex['green'] = rgbs[:, 1]
        vertex['blue'] = rgbs[:, 2]
        
        el = PlyElement.describe(vertex, 'vertex')
        ply_out_path = os.path.join(os.path.dirname(args.out), "points3d.ply")
        PlyData([el], text=True).write(ply_out_path)
        print(f"Saved {len(xyzs)} sparse points to {ply_out_path}")
    except Exception as e:
        print(f"Warning: Could not read or save points3D.bin: {e}")

    frames = []
    
    # For simplicity, assuming all images use the same camera (intrinsic)
    # or just picking the first one's FOV.
    cam_id = list(cam_intrinsics.keys())[0]
    intr = cam_intrinsics[cam_id]
    
    # Assuming PINHOLE model: [f_x, f_y, c_x, c_y]
    if intr.model == "PINHOLE":
        fl_x = intr.params[0]
        fl_y = intr.params[1]
        cx = intr.params[2]
        cy = intr.params[3]
    elif intr.model == "SIMPLE_PINHOLE":
        fl_x = intr.params[0]
        fl_y = intr.params[0]
        cx = intr.params[1]
        cy = intr.params[2]
    else:
        # Fallback
        fl_x = intr.params[0]
        fl_y = intr.params[0]
    
    camera_angle_x = math.atan(intr.width / (fl_x * 2)) * 2
    
    for key, extr in cam_extrinsics.items():
        # Extrinsics is World-to-Camera
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        
        # Convert to Camera-to-World
        c2w = np.linalg.inv(w2c)
        
        # COLMAP is right-down-forward, NeRF is right-up-back
        # We need to flip Y and Z axes
        c2w[0:3, 1:3] *= -1
        
        # Remove extension from filename for the json format
        file_path = "input/" + os.path.splitext(extr.name)[0]
        
        frame = {
            "file_path": file_path,
            "transform_matrix": c2w.tolist()
        }
        frames.append(frame)
        
    out_dict = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }
    
    with open(args.out, 'w') as f:
        json.dump(out_dict, f, indent=4)
        
    print(f"Successfully converted {len(frames)} frames to {args.out}")

if __name__ == "__main__":
    main()
