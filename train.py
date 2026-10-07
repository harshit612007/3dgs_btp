import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. MATHEMATICAL UTILITIES
# ==========================================
def q_to_rot_matrix(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y**2 + z**2), 2 * (x*y - w*z),     2 * (x*z + w*y),
        2 * (x*y + w*z),       1 - 2 * (x**2 + z**2), 2 * (y*z - w*x),
        2 * (x*z - w*y),       2 * (y*z + w*x),       1 - 2 * (x**2 + y**2)
    ], dim=1).reshape(-1, 3, 3)
    return R

def compute_cov3d(scale, rot):
    S = torch.zeros((scale.shape[0], 3, 3), device=scale.device)
    S[:, 0, 0] = scale[:, 0]
    S[:, 1, 1] = scale[:, 1]
    S[:, 2, 2] = scale[:, 2]
    R = q_to_rot_matrix(rot)
    M = torch.bmm(R, S)
    return torch.bmm(M, M.transpose(1, 2))

def project_cov3d_to_cov2d(cov3d, view_pos, focal_x, focal_y, W, H):
    x, y, z = view_pos[:, 0], view_pos[:, 1], view_pos[:, 2]
    z = torch.clamp(z, min=1e-3)
    
    J = torch.zeros((view_pos.shape[0], 2, 3), device=view_pos.device)
    J[:, 0, 0] = focal_x / z
    J[:, 0, 2] = -(focal_x * x) / (z**2)
    J[:, 1, 1] = focal_y / z
    J[:, 1, 2] = -(focal_y * y) / (z**2)
    
    cov2d = torch.bmm(J, torch.bmm(cov3d, J.transpose(1, 2)))
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3
    return cov2d

# ==========================================
# 2. SCENE REPRESENTATION (EXPLICIT)
# ==========================================
class GaussianModel(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.xyz = nn.Parameter(torch.rand(num_points, 3) * 2 - 1)
        self.features_dc = nn.Parameter(torch.rand(num_points, 3)) # RGB
        self.scaling = nn.Parameter(torch.log(torch.rand(num_points, 3) * 0.1 + 0.01))
        self.rotation = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(num_points, 1))
        self.opacity = nn.Parameter(torch.zeros(num_points, 1))

    def get_xyz(self): return self.xyz
    def get_features(self): return torch.sigmoid(self.features_dc)
    def get_scaling(self): return torch.clamp(torch.exp(self.scaling), max=10.0)
    def get_rotation(self): return self.rotation
    def get_opacity(self): return torch.sigmoid(self.opacity)

    def save_ply(self, path):
        # 1. Opacity Culling (Delete invisible points to save space!)
        opacities = self.get_opacity().detach().cpu()
        mask = (opacities > 0.01).squeeze()
        
        xyz = self.xyz.detach().cpu()[mask].numpy()
        # 2. Float16 Quantization for colors
        f_dc = self.get_features().detach().cpu()[mask].half().float().numpy() 
        opacity = opacities[mask].numpy()
        scale = self.get_scaling().detach().cpu()[mask].numpy()
        rot = self.rotation.detach().cpu()[mask].numpy()
        
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                 ('opacity', 'f4'),
                 ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                 ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
        
        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, f_dc, opacity, scale, rot), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        with open(path, 'wb') as f:
            f.write(b"ply\nformat binary_little_endian 1.0\n")
            f.write(f"element vertex {xyz.shape[0]}\n".encode())
            f.write(b"property float x\nproperty float y\nproperty float z\n")
            f.write(b"property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n")
            f.write(b"property float opacity\n")
            f.write(b"property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
            f.write(b"property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
            f.write(b"end_header\n")
            f.write(elements.tobytes())
            
    def load_ply(self, path):
        with open(path, 'rb') as f:
            content = f.read()
        
        header_end = content.find(b'end_header\n') + 11
        vertex_line = [line for line in content[:header_end].split(b'\n') if b'element vertex' in line][0]
        num_points = int(vertex_line.split()[2])
        
        dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                 ('opacity', 'f4'),
                 ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                 ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')])
        
        elements = np.frombuffer(content[header_end:], dtype=dtype)
        
        # We need to manually set the parameters using the inverse of the activation functions
        self.xyz = nn.Parameter(torch.tensor(np.stack((elements['x'], elements['y'], elements['z']), axis=-1)).float())
        self.features_dc = nn.Parameter(torch.tensor(np.stack((elements['f_dc_0'], elements['f_dc_1'], elements['f_dc_2']), axis=-1)).float())
        
        # Inverse sigmoid for opacity: -ln(1/y - 1)
        op = np.clip(elements['opacity'], 1e-4, 1.0 - 1e-4)
        self.opacity = nn.Parameter(torch.tensor(-np.log(1.0/op - 1.0)).unsqueeze(1).float())
        
        # Inverse exp for scaling
        sc = np.clip(np.stack((elements['scale_0'], elements['scale_1'], elements['scale_2']), axis=-1), 1e-4, None)
        self.scaling = nn.Parameter(torch.tensor(np.log(sc)).float())
        
        self.rotation = nn.Parameter(torch.tensor(np.stack((elements['rot_0'], elements['rot_1'], elements['rot_2'], elements['rot_3']), axis=-1)).float())

        print(f"Loaded optimized point cloud with {num_points} Gaussians from {path}")

# ==========================================
# 3. PURE PYTORCH RASTERIZER
# ==========================================
def render(camera_info, model: GaussianModel, bg_color):
    W, H = camera_info['width'], camera_info['height']
    fx, fy, cx, cy = camera_info['fx'], camera_info['fy'], camera_info['cx'], camera_info['cy']
    w2c = camera_info['w2c']
    
    xyz = model.get_xyz()
    color, scale = model.get_features(), model.get_scaling()
    rot, opacity = model.get_rotation(), model.get_opacity()
    
    xyz_homo = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device=device)], dim=1)
    view_pos = torch.matmul(w2c, xyz_homo.T).T[:, :3]
    
    valid_mask = view_pos[:, 2] > 0.1
    if valid_mask.sum() == 0: return bg_color.view(3, 1, 1).expand(3, H, W) + (model.get_xyz()[0,0] * 0.0) # Dummy grad link
        
    view_pos, color, scale, rot, opacity = view_pos[valid_mask], color[valid_mask], scale[valid_mask], rot[valid_mask], opacity[valid_mask]
    
    z = view_pos[:, 2]
    u = (view_pos[:, 0] * fx / z) + cx
    v = (view_pos[:, 1] * fy / z) + cy
    uv = torch.stack([u, v], dim=1)
    
    screen_mask = (u > -W) & (u < 2*W) & (v > -H) & (v < 2*H)
    if screen_mask.sum() == 0: return bg_color.view(3, 1, 1).expand(3, H, W) + (model.get_xyz()[0,0] * 0.0) # Dummy grad link
        
    view_pos, uv, color, scale, rot, opacity, z = view_pos[screen_mask], uv[screen_mask], color[screen_mask], scale[screen_mask], rot[screen_mask], opacity[screen_mask], z[screen_mask]
    
    cov3d = compute_cov3d(scale, rot)
    cov2d = project_cov3d_to_cov2d(cov3d, view_pos, fx, fy, W, H)
    
    det = torch.clamp(cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0], min=1e-5)
    
    inv_cov2d = torch.zeros_like(cov2d)
    inv_cov2d[:, 0, 0], inv_cov2d[:, 1, 1] = cov2d[:, 1, 1] / det, cov2d[:, 0, 0] / det
    inv_cov2d[:, 0, 1], inv_cov2d[:, 1, 0] = -cov2d[:, 0, 1] / det, -cov2d[:, 1, 0] / det
    
    sorted_indices = torch.argsort(z, descending=False)
    uv, inv_cov2d, opacity, color = uv[sorted_indices], inv_cov2d[sorted_indices], opacity[sorted_indices], color[sorted_indices]
    
    # Force grid to be float32 to prevent float16 overflow (inf/NaN) at resolutions > 256
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack([x_grid, y_grid], dim=-1).float()
    
    out_color = torch.zeros((H, W, 3), device=device, dtype=torch.float32)
    transmittance = torch.ones((H, W, 1), device=device, dtype=torch.float32)
    
    chunk_size = 512
    
    # Define checkpointable function to completely delete float32 memory overhead during forward pass
    @torch.compile(mode="reduce-overhead")
    def compute_chunk(mu, inv_cov, op, c, trans):
        # Fully vectorized dense math (Maximizes GPU memory bandwidth instead of CPU kernel overhead)
        dx = grid[:,:,0].unsqueeze(2) - mu[:, 0].view(1, 1, -1)
        dy = grid[:,:,1].unsqueeze(2) - mu[:, 1].view(1, 1, -1)
        dist2 = dx*dx*inv_cov[:,0,0].view(1,1,-1) + 2*dx*dy*inv_cov[:,0,1].view(1,1,-1) + dy*dy*inv_cov[:,1,1].view(1,1,-1)
        
        alpha_chunk = torch.exp(-0.5 * dist2.half()) * (dist2.half() < 16.0).half() * op.view(1, 1, -1)
            
        T = torch.cat([torch.ones(H, W, 1, device=device, dtype=torch.half), torch.cumprod(torch.clamp(1.0 - alpha_chunk[:, :, :-1], min=1e-4), dim=2)], dim=2)
        weight = T * alpha_chunk
        
        color_upd = (trans.half() * torch.matmul(weight, c)).float()
        trans_upd = torch.prod(torch.clamp(1.0 - alpha_chunk.float(), min=1e-4), dim=2, keepdim=True)
        return color_upd, trans_upd

    import torch.utils.checkpoint as checkpoint
    
    for i in range(0, view_pos.shape[0], chunk_size):
        end = min(i + chunk_size, view_pos.shape[0])
        
        mu_chunk = uv[i:end]
        inv_cov_chunk = inv_cov2d[i:end]
        op_chunk = opacity[i:end].half()
        c_chunk = color[i:end].half()
        
        # PyTorch will NOT store dx, dy, dist2, alpha, T, or weight in VRAM!
        color_update, trans_update = checkpoint.checkpoint(compute_chunk, mu_chunk, inv_cov_chunk, op_chunk, c_chunk, transmittance, use_reentrant=False)
        
        out_color = out_color + color_update
        transmittance = transmittance * trans_update
            
    return (out_color + transmittance * bg_color.view(1, 1, 3)).permute(2, 0, 1)

def ssim(img1, img2):
    mu1 = F.avg_pool2d(img1.unsqueeze(0), 11, 1, 5)
    mu2 = F.avg_pool2d(img2.unsqueeze(0), 11, 1, 5)
    sigma1_sq = F.avg_pool2d(img1.unsqueeze(0)**2, 11, 1, 5) - mu1**2
    sigma2_sq = F.avg_pool2d(img2.unsqueeze(0)**2, 11, 1, 5) - mu2**2
    sigma12 = F.avg_pool2d(img1.unsqueeze(0)*img2.unsqueeze(0), 11, 1, 5) - mu1*mu2
    C1, C2 = 0.0001, 0.0009
    return (((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))).mean()

# ==========================================
# 4. DATA LOADING
# ==========================================
def load_dataset(path, target_size=128, white_background=False):
    json_path = os.path.join(path, "transforms_train.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(path, "transforms.json") # Fallback for real-world scenes
    with open(json_path, 'r') as f: meta = json.load(f)
    cameras = []
    print(f"Loading {len(meta['frames'])} images from {path}...")
    for frame in meta['frames']:
        base_path = os.path.join(path, frame['file_path'])
        img_path = next((base_path + ext for ext in ['.png', '.jpeg', '.jpg'] if os.path.exists(base_path + ext)), base_path if os.path.exists(base_path) else None)
        if not img_path: continue
        
        image = Image.open(img_path)
        if image.mode == 'RGBA':
            bg = Image.new('RGB', image.size, (255, 255, 255) if white_background else (0, 0, 0))
            bg.paste(image, mask=image.split()[3])
            image = bg
        else:
            image = image.convert("RGB")
        orig_W, orig_H = image.size
        img_tensor = torch.from_numpy(np.array(image.resize((target_size, target_size), Image.Resampling.BILINEAR))).float() / 255.0
        
        c2w = torch.tensor(frame['transform_matrix']).float()
        c2w[:, 1:3] *= -1 # Coordinate system fix
        
        fx = (.5 * orig_W / np.tan(.5 * meta.get('camera_angle_x', math.pi/2.0))) * (target_size / orig_W)
        cameras.append({
            'w2c': torch.linalg.inv(c2w).to(device), 'width': target_size, 'height': target_size,
            'fx': fx, 'fy': fx, 'cx': target_size / 2.0, 'cy': target_size / 2.0,
            'gt_image': img_tensor.to(device).permute(2, 0, 1), 'c2w': c2w.to(device)
        })
    return cameras

# ==========================================
# 5. NOVEL VIEW SYNTHESIS (3D VIDEO)
# ==========================================
def render_trajectory(model, cameras, output_dir, num_frames=60):
    """Renders a smooth 3D video fly-through for the research paper evaluation."""
    print(f"Rendering 3D Trajectory Video ({num_frames} frames)...")
    video_dir = os.path.join(output_dir, "video_frames")
    os.makedirs(video_dir, exist_ok=True)
    
    # Object is at the origin (0, 0, 0)
    scene_center = torch.zeros(3, device=device)
    centers = torch.stack([cam['c2w'][:3, 3] for cam in cameras])
    
    # Radius is the average distance from origin to original cameras
    radius = torch.norm(centers, dim=1).mean()
    
    # We want to orbit at the average Z-height of the original cameras
    z_height = centers[:, 2].mean()
    
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)
    cam = cameras[0] # Use intrinsic parameters of first camera
    
    for i in range(num_frames):
        angle = (i / num_frames) * 2 * math.pi
        
        # Spiral motion (Orbit in X-Y plane around the origin)
        cam_x = radius * math.cos(angle)
        cam_y = radius * math.sin(angle)
        cam_z = z_height + math.sin(angle * 2) * (radius * 0.1)
        
        c2w = torch.eye(4, device=device)
        c2w[0, 3], c2w[1, 3], c2w[2, 3] = cam_x, cam_y, cam_z
        
        # Look exactly at the bulldozer (the origin)
        forward = F.normalize(scene_center - c2w[:3, 3], dim=0)
        up_world = torch.tensor([0.0, 0.0, 1.0], device=device)
        right = F.normalize(torch.cross(forward, up_world), dim=0)
        down = F.normalize(torch.cross(forward, right), dim=0) # OpenCV Y is down
        
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = right, down, forward
        
        test_cam = cam.copy()
        test_cam['w2c'] = torch.linalg.inv(c2w)
        
        with torch.no_grad():
            rendered = render(test_cam, model, bg_color)
            out_img = (rendered.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(out_img).save(os.path.join(video_dir, f"frame_{i:04d}.png"))
    print(f"Trajectory saved to {video_dir}! You can stitch these into an MP4.")

# ==========================================
# 6. MAIN TRAINING LOOP
# ==========================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cameras = load_dataset(args.dataset_path, args.resolution, args.white_background)
    
    model = GaussianModel(args.num_points).to(device)
    optimizer = torch.optim.Adam([
        {'params': [model.xyz], 'lr': 1.6e-4},
        {'params': [model.features_dc], 'lr': 0.0025},
        {'params': [model.opacity], 'lr': 0.05},
        {'params': [model.scaling], 'lr': 0.005},
        {'params': [model.rotation], 'lr': 0.001}
    ])
    
    # Official 3DGS schedule: Only decay xyz (position) learning rate
    def xyz_decay(step):
        return (0.01) ** (step / args.iterations)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=[xyz_decay, lambda _: 1.0, lambda _: 1.0, lambda _: 1.0, lambda _: 1.0]
    )
    
    # Mixed Precision Scaler for VRAM savings
    scaler = torch.cuda.amp.GradScaler()
    
    bg_color = torch.tensor([1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0], device=device)
    
    best_loss = float('inf')
    best_img_path = None
    
    print(f"Starting training for {args.iterations} iterations...")
    start_time = time.time()
    
    for i in range(args.iterations):
        optimizer.zero_grad()
        
        cam = cameras[np.random.randint(0, len(cameras))]
        
        rendered_img = render(cam, model, bg_color)
        loss = 0.8 * F.l1_loss(rendered_img, cam['gt_image']) + 0.2 * (1.0 - ssim(rendered_img, cam['gt_image']))
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if i % 10 == 0:
            if i > 0:
                elapsed = time.time() - start_time
                eta_seconds = (elapsed / i) * (args.iterations - i)
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = "Calculating..."
            
            print(f"Iter {i:05d}/{args.iterations} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | ETA: {eta_str}")
            
        if (i+1) % args.save_freq == 0 or i == args.iterations - 1:
            # Render a FIXED camera (cameras[0]) so we can compare apples-to-apples visually
            with torch.no_grad():
                fixed_cam = cameras[0]
                eval_img = render(fixed_cam, model, bg_color)
                out_img = (eval_img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                save_path = os.path.join(args.output_dir, f"render_{i}.png")
                Image.fromarray(out_img).save(save_path)
                print(f"Saved progress image: {save_path}")
            
    print("Training complete!")
    ply_path = os.path.join(args.output_dir, "optimized_scene.ply")
    model.save_ply(ply_path)
    
    if args.render_video:
        render_trajectory(model, cameras, args.output_dir, num_frames=args.video_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hardware-Agnostic PyTorch 3DGS")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset (e.g. datasets/tiny_dataset)")
    parser.add_argument("--output_dir", type=str, default="outputs/experiment_1", help="Where to save renders")
    parser.add_argument("--iterations", type=int, default=300, help="Number of training iterations")
    parser.add_argument("--num_points", type=int, default=1000, help="Number of 3D Gaussians")
    parser.add_argument("--resolution", type=int, default=128, help="Resolution to train at (lower is faster)")
    parser.add_argument("--save_freq", type=int, default=50, help="Save a training snapshot every X iterations")
    parser.add_argument("--render_video", action="store_true", help="Generate a 360-degree flythrough video at the end")
    parser.add_argument("--video_frames", type=int, default=60, help="Number of frames in the 360-degree video")
    parser.add_argument("--white_background", action="store_true", help="Train on a white background instead of black (for datasets with alpha channels)")
    
    args = parser.parse_args()
    main(args)
