import os
import torch
import numpy as np
from PIL import Image
from plyfile import PlyData
from pure_pytorch_3dgs import load_dataset, render, GaussianModel
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ply_into_model(path):
    ply = PlyData.read(path)
    pts = ply.elements[0].data
    
    xyz = np.stack((pts['x'], pts['y'], pts['z']), axis=-1)
    f_dc = np.stack((pts['f_dc_0'], pts['f_dc_1'], pts['f_dc_2']), axis=-1)
    opacity = pts['opacity']
    scale = np.stack((pts['scale_0'], pts['scale_1'], pts['scale_2']), axis=-1)
    rot = np.stack((pts['rot_0'], pts['rot_1'], pts['rot_2'], pts['rot_3']), axis=-1)
    
    model = GaussianModel(xyz.shape[0]).to(device)
    with torch.no_grad():
        model.xyz.copy_(torch.tensor(xyz, dtype=torch.float32, device=device))
        model.features_dc.copy_(torch.tensor(f_dc, dtype=torch.float32, device=device))
        model.opacity.copy_(torch.tensor(opacity, dtype=torch.float32, device=device).unsqueeze(1))
        model.scaling.copy_(torch.tensor(scale, dtype=torch.float32, device=device))
        model.rotation.copy_(torch.tensor(rot, dtype=torch.float32, device=device))
        
    return model

def main():
    print("Loading cameras...")
    cameras = load_dataset("C:/Users/HARSHIT/Documents/3dgs/tiny_dataset")
    
    print("Loading optimized model...")
    ply_path = "C:/Users/HARSHIT/Documents/3dgs/output_tiny/point_cloud.ply"
    model = load_ply_into_model(ply_path)
    
    bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)
    
    # Render the first camera view
    cam = cameras[0]
    rendered_img = render(cam, model, bg_color)
    
    # Convert to image
    out_img = (rendered_img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_path = "C:/Users/HARSHIT/Documents/3dgs/output_tiny/final_render.png"
    Image.fromarray(out_img).save(img_path)
    
    print(f"Saved final render to {img_path}")

if __name__ == "__main__":
    main()
