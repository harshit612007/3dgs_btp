import os
import argparse
import torch
import numpy as np
import time
from plyfile import PlyData
from pure_pytorch_3dgs import load_dataset, render, ssim, GaussianModel
from export_ply import save_ply
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def init_from_ply(path):
    ply = PlyData.read(path)
    pts = ply.elements[0].data
    
    # Randomly downsample if there are too many points to avoid OOM
    # Pure PyTorch requires massive memory for N * H * W
    MAX_POINTS = 15000
    if len(pts) > MAX_POINTS:
        print(f"Downsampling {len(pts)} points to {MAX_POINTS} to avoid OOM...")
        indices = np.random.choice(len(pts), MAX_POINTS, replace=False)
        pts = pts[indices]
        
    xyz = np.stack((pts['x'], pts['y'], pts['z']), axis=-1)
    
    model = GaussianModel(xyz.shape[0]).to(device)
    
    with torch.no_grad():
        model.xyz.copy_(torch.tensor(xyz, dtype=torch.float32, device=device))
        
        # Initialize color from ply
        if 'red' in pts.dtype.names:
            rgb = np.stack((pts['red'], pts['green'], pts['blue']), axis=-1) / 255.0
            # Simple DC initialization
            f_dc = (rgb - 0.5) / 0.28209
            model.features_dc.copy_(torch.tensor(f_dc, dtype=torch.float32, device=device))
            
        # Initialize scale based on nearest neighbor distances
        # For simplicity, we just use a small constant scale
        model.scaling.copy_(torch.ones_like(model.scaling) * -5.0) 
        
        # Initialize opacity to 0.1
        model.opacity.copy_(torch.ones_like(model.opacity) * -2.197) # inverse sigmoid of 0.1
        
    return model

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train 3DGS model")
    parser.add_argument("--source", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--iters", type=int, default=200, help="Number of training iterations")
    parser.add_argument("--out", type=str, default="output", help="Path to the output directory")
    args = parser.parse_args()

    print(f"Loading dataset from {args.source}...")
    cameras = load_dataset(args.source)
    print(f"Loaded {len(cameras)} cameras.")
    
    # Check if points3D.ply exists, otherwise initialize randomly
    ply_path = os.path.join(args.source, "points3d.ply")
    if os.path.exists(ply_path):
        model = init_from_ply(ply_path)
    else:
        print("points3d.ply not found. Initializing with random points...")
        model = GaussianModel(10000).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': [model.xyz], 'lr': 0.00016},
        {'params': [model.features_dc], 'lr': 0.0025},
        {'params': [model.opacity], 'lr': 0.05},
        {'params': [model.scaling], 'lr': 0.005},
        {'params': [model.rotation], 'lr': 0.001}
    ])
    
    bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)
    
    iterations = args.iters
    print(f"Starting optimization for {iterations} iterations...")
    
    start_time = time.time()
    for iter in range(1, iterations + 1):
        cam_idx = np.random.randint(0, len(cameras))
        cam = cameras[cam_idx]
        
        optimizer.zero_grad()
        
        rendered_img = render(cam, model, bg_color)
        
        gt_img = cam['gt_image'].to(device)
        
        loss = F.l1_loss(rendered_img, gt_img)
        
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
            
        if iter % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Iteration {iter}/{iterations}, Loss: {loss.item():.4f}, Time/10iters: {elapsed:.2f}s")
            start_time = time.time()
            
    print("Optimization finished! Exporting scene...")
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "point_cloud.ply")
    save_ply(model, out_path)
    print(f"Saved reconstructed scene to {out_path}")

if __name__ == "__main__":
    main()
