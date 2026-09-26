import os
import json
import math
import numpy as np
from PIL import Image

def get_camera_matrix(radius, theta, phi):
    # theta: polar angle, phi: azimuthal angle
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)
    
    forward = np.array([-x, -y, -z])
    forward = forward / np.linalg.norm(forward)
    
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # NeRF convention: camera looks down -Z, Y is up, X is right
    # However, Blender convention: camera looks down -Z, Y is up
    # Just construct a 4x4 matrix
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = [x, y, z]
    return c2w.tolist()

def generate_dataset(output_dir="tiny_dataset"):
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    frames_train = []
    frames_test = []
    
    num_train = 20
    num_test = 5
    radius = 4.0
    
    print("Generating images...")
    for i in range(num_train + num_test):
        is_train = i < num_train
        
        # Simple color based on angle
        phi = (i / (num_train + num_test)) * 2 * math.pi
        theta = math.pi / 3 # 60 degrees down
        
        c2w = get_camera_matrix(radius, theta, phi)
        
        # Draw a simple image (e.g. solid color + gradient)
        img = np.zeros((400, 400, 4), dtype=np.uint8)
        
        # Add a cube in the center
        # Since it's a dummy dataset, we don't need real raytracing, just an image that Scaffold-GS can load
        img[:, :] = [50 + int(200 * math.sin(phi)**2), 100, 200, 255]
        
        # Center square
        img[150:250, 150:250] = [255, 0, 0, 255]
        
        im = Image.fromarray(img)
        
        if is_train:
            filename = f"train/r_{i}.png"
            im.save(os.path.join(output_dir, filename))
            frames_train.append({
                "file_path": f"./{filename.split('.')[0]}",
                "rotation": 0.0,
                "transform_matrix": c2w
            })
        else:
            filename = f"test/r_{i}.png"
            im.save(os.path.join(output_dir, filename))
            frames_test.append({
                "file_path": f"./{filename.split('.')[0]}",
                "rotation": 0.0,
                "transform_matrix": c2w
            })
            
    # Save transforms
    camera_angle_x = 0.69
    
    with open(os.path.join(output_dir, "transforms_train.json"), "w") as f:
        json.dump({"camera_angle_x": camera_angle_x, "frames": frames_train}, f, indent=4)
        
    with open(os.path.join(output_dir, "transforms_test.json"), "w") as f:
        json.dump({"camera_angle_x": camera_angle_x, "frames": frames_test}, f, indent=4)

    print(f"Dataset generated at {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    generate_dataset()
