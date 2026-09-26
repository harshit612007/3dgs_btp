import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def q_to_rot_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix"""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R = torch.stack([
        1 - 2 * (y**2 + z**2), 2 * (x*y - w*z),     2 * (x*z + w*y),
        2 * (x*y + w*z),       1 - 2 * (x**2 + z**2), 2 * (y*z - w*x),
        2 * (x*z - w*y),       2 * (y*z + w*x),       1 - 2 * (x**2 + y**2)
    ], dim=1).reshape(-1, 3, 3)
    return R

def compute_cov3d(scale, rot):
    """Compute 3D covariance matrix from scale and rotation"""
    S = torch.zeros((scale.shape[0], 3, 3), device=scale.device)
    S[:, 0, 0] = scale[:, 0]
    S[:, 1, 1] = scale[:, 1]
    S[:, 2, 2] = scale[:, 2]
    
    R = q_to_rot_matrix(rot)
    M = torch.bmm(R, S)
    Cov3D = torch.bmm(M, M.transpose(1, 2))
    return Cov3D

def project_cov3d_to_cov2d(cov3d, view_pos, focal_x, focal_y, W, H):
    """Project 3D covariance to 2D covariance using Jacobian of perspective projection"""
    x, y, z = view_pos[:, 0], view_pos[:, 1], view_pos[:, 2]
    
    # Avoid division by zero
    z = torch.clamp(z, min=1e-3)
    
    J = torch.zeros((view_pos.shape[0], 2, 3), device=view_pos.device)
    J[:, 0, 0] = focal_x / z
    J[:, 0, 2] = -(focal_x * x) / (z**2)
    J[:, 1, 1] = focal_y / z
    J[:, 1, 2] = -(focal_y * y) / (z**2)
    
    cov2d = torch.bmm(J, torch.bmm(cov3d, J.transpose(1, 2)))
    
    # Add a small offset to diagonal for numerical stability (low pass filter)
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3
    
    return cov2d

class GaussianModel(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        # Initialize randomly in a small cube
        self.xyz = nn.Parameter(torch.rand(num_points, 3) * 2 - 1)
        self.features_dc = nn.Parameter(torch.rand(num_points, 3)) # RGB
        self.scaling = nn.Parameter(torch.log(torch.rand(num_points, 3) * 0.1 + 0.01))
        self.rotation = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(num_points, 1))
        self.opacity = nn.Parameter(torch.zeros(num_points, 1)) # Inverse sigmoid

    def get_xyz(self): return self.xyz
    def get_features(self): return torch.sigmoid(self.features_dc)
    def get_scaling(self): return torch.exp(self.scaling)
    def get_rotation(self): return self.rotation
    def get_opacity(self): return torch.sigmoid(self.opacity)

def render(camera_info, model: GaussianModel, bg_color):
    """
    Pure PyTorch differentiable renderer.
    """
    W, H = camera_info['width'], camera_info['height']
    fx, fy = camera_info['fx'], camera_info['fy']
    cx, cy = camera_info['cx'], camera_info['cy']
    w2c = camera_info['w2c'] # 4x4 matrix
    
    xyz = model.get_xyz()
    color = model.get_features()
    scale = model.get_scaling()
    rot = model.get_rotation()
    opacity = model.get_opacity()
    
    num_points = xyz.shape[0]
    
    # 1. Transform xyz to camera space
    xyz_homo = torch.cat([xyz, torch.ones((num_points, 1), device=device)], dim=1)
    view_pos_homo = torch.matmul(w2c, xyz_homo.T).T
    view_pos = view_pos_homo[:, :3]
    
    # 2. Filter points behind the camera
    valid_mask = view_pos[:, 2] > 0.1
    if valid_mask.sum() == 0:
        return bg_color.view(3, 1, 1).expand(3, H, W)
        
    view_pos = view_pos[valid_mask]
    xyz_valid = xyz[valid_mask]
    color_valid = color[valid_mask]
    scale_valid = scale[valid_mask]
    rot_valid = rot[valid_mask]
    opacity_valid = opacity[valid_mask]
    
    num_valid = view_pos.shape[0]
    
    # 3. Project to 2D
    z = view_pos[:, 2]
    u = (view_pos[:, 0] * fx / z) + cx
    v = (view_pos[:, 1] * fy / z) + cy
    uv = torch.stack([u, v], dim=1)
    
    # 4. Filter points far outside the image
    screen_mask = (u > -W) & (u < 2*W) & (v > -H) & (v < 2*H)
    if screen_mask.sum() == 0:
        return bg_color.view(3, 1, 1).expand(3, H, W)
        
    view_pos = view_pos[screen_mask]
    uv = uv[screen_mask]
    color_valid = color_valid[screen_mask]
    scale_valid = scale_valid[screen_mask]
    rot_valid = rot_valid[screen_mask]
    opacity_valid = opacity_valid[screen_mask]
    z = z[screen_mask]
    num_final = view_pos.shape[0]
    
    # 5. Compute 2D Covariance
    cov3d = compute_cov3d(scale_valid, rot_valid)
    cov2d = project_cov3d_to_cov2d(cov3d, view_pos, fx, fy, W, H)
    
    # Invert covariance
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    det = torch.clamp(det, min=1e-5)
    inv_cov2d = torch.zeros_like(cov2d)
    inv_cov2d[:, 0, 0] = cov2d[:, 1, 1] / det
    inv_cov2d[:, 1, 1] = cov2d[:, 0, 0] / det
    inv_cov2d[:, 0, 1] = -cov2d[:, 0, 1] / det
    inv_cov2d[:, 1, 0] = -cov2d[:, 1, 0] / det
    
    # 6. Sort points by depth (front to back)
    sorted_indices = torch.argsort(z, descending=False)
    
    uv_sorted = uv[sorted_indices]
    inv_cov2d_sorted = inv_cov2d[sorted_indices]
    opacity_sorted = opacity_valid[sorted_indices]
    color_sorted = color_valid[sorted_indices]
    
    # 7. Evaluate Gaussian PDFs and composite on grid
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack([x_grid.float(), y_grid.float()], dim=-1) # [H, W, 2]
    
    out_color = torch.zeros((H, W, 3), device=device)
    transmittance = torch.ones((H, W, 1), device=device)
    
    chunk_size = 512
    for i in range(0, num_final, chunk_size):
        end = min(i + chunk_size, num_final)
        
        mu_chunk = uv_sorted[i:end]
        inv_cov_chunk = inv_cov2d_sorted[i:end]
        op_chunk = opacity_sorted[i:end]
        c_chunk = color_sorted[i:end]
        
        dx = grid[:,:,0].unsqueeze(2) - mu_chunk[:, 0].view(1, 1, -1)
        dy = grid[:,:,1].unsqueeze(2) - mu_chunk[:, 1].view(1, 1, -1)
        
        c00 = inv_cov_chunk[:, 0, 0].view(1, 1, -1)
        c01 = inv_cov_chunk[:, 0, 1].view(1, 1, -1)
        c11 = inv_cov_chunk[:, 1, 1].view(1, 1, -1)
        
        dist2 = dx * dx * c00 + 2 * dx * dy * c01 + dy * dy * c11
        
        alpha = torch.exp(-0.5 * dist2) # [H, W, chunk]
        
        mask = dist2 < 16.0
        alpha = alpha * mask.float()
        
        alpha = alpha * op_chunk.view(1, 1, -1) # [H, W, chunk]
        
        T = torch.cat([torch.ones(H, W, 1, device=device), torch.cumprod(1.0 - alpha[:, :, :-1], dim=2)], dim=2)
        T_global = transmittance * T
        weights = T_global * alpha
        out_color = out_color + (weights.unsqueeze(-1) * c_chunk.view(1, 1, -1, 3)).sum(dim=2)
        transmittance = transmittance * torch.prod(1.0 - alpha, dim=2, keepdim=True)
            
    out_color = out_color + transmittance * bg_color.view(1, 1, 3)
    return out_color.permute(2, 0, 1)

def ssim(img1, img2):
    mu1 = F.avg_pool2d(img1.unsqueeze(0), 11, 1, 5)
    mu2 = F.avg_pool2d(img2.unsqueeze(0), 11, 1, 5)
    sigma1_sq = F.avg_pool2d(img1.unsqueeze(0)**2, 11, 1, 5) - mu1**2
    sigma2_sq = F.avg_pool2d(img2.unsqueeze(0)**2, 11, 1, 5) - mu2**2
    sigma12 = F.avg_pool2d(img1.unsqueeze(0)*img2.unsqueeze(0), 11, 1, 5) - mu1*mu2
    
    C1 = (0.01)**2
    C2 = (0.03)**2
    
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def load_dataset(path):
    with open(os.path.join(path, "transforms_train.json"), 'r') as f:
        meta = json.load(f)
        
    cameras = []
    
    TARGET_SIZE = 128 # Smaller to run fast
    
    print(f"Loading {len(meta['frames'])} images...")
    for idx, frame in enumerate(meta['frames']):
        if idx % 10 == 0:
            print(f"Loaded {idx}/{len(meta['frames'])} images...")
            
        base_path = os.path.join(path, frame['file_path'])
        
        # Try different extensions
        img_path = None
        for ext in ['.png', '.jpeg', '.jpg', '.JPG', '.JPEG']:
            if os.path.exists(base_path + ext):
                img_path = base_path + ext
                break
        
        if img_path is None:
            # Maybe the extension is already in file_path
            if os.path.exists(base_path):
                img_path = base_path
            else:
                continue
            
        image = Image.open(img_path).convert("RGB")
        orig_W, orig_H = image.size
        
        image = image.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BILINEAR)
        img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        img_tensor = img_tensor.to(device).permute(2, 0, 1)
        
        c2w = torch.tensor(frame['transform_matrix']).float()
        
        c2w[:, 1:3] *= -1
        
        w2c = torch.linalg.inv(c2w).to(device)
        
        camera_angle_x = meta.get('camera_angle_x', math.pi/2.0)
        fx = .5 * orig_W / np.tan(.5 * camera_angle_x)
        fy = fx
        
        fx = fx * (TARGET_SIZE / orig_W)
        fy = fy * (TARGET_SIZE / orig_H)
        
        cam = {
            'w2c': w2c,
            'width': TARGET_SIZE,
            'height': TARGET_SIZE,
            'fx': fx,
            'fy': fy,
            'cx': TARGET_SIZE / 2.0,
            'cy': TARGET_SIZE / 2.0,
            'gt_image': img_tensor
        }
        cameras.append(cam)
        
    return cameras

def main():
    print("Loading dataset...")
    cameras = load_dataset("C:/Users/HARSHIT/Documents/3dgs/tiny_dataset")
    print(f"Loaded {len(cameras)} cameras.")
    
    num_points = 500
    model = GaussianModel(num_points).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': [model.xyz], 'lr': 0.005},
        {'params': [model.features_dc], 'lr': 0.01},
        {'params': [model.opacity], 'lr': 0.05},
        {'params': [model.scaling], 'lr': 0.005},
        {'params': [model.rotation], 'lr': 0.005}
    ])
    
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)
    
    iterations = 200
    print("Starting optimization...")
    
    os.makedirs("C:/Users/HARSHIT/Documents/3dgs/pure_output", exist_ok=True)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        cam_idx = np.random.randint(0, len(cameras))
        cam = cameras[cam_idx]
        
        rendered_img = render(cam, model, bg_color)
        
        gt = cam['gt_image']
        l1 = F.l1_loss(rendered_img, gt)
        l_ssim = 1.0 - ssim(rendered_img, gt)
        
        loss = 0.8 * l1 + 0.2 * l_ssim
        
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Iteration {i}/{iterations} - Loss: {loss.item():.4f}")
            
        if (i+1) % 50 == 0 or i == iterations - 1:
            out_img = (rendered_img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(out_img).save(f"C:/Users/HARSHIT/Documents/3dgs/pure_output/render_{i}.png")
            print(f"Saved render_{i}.png")

if __name__ == "__main__":
    main()
