import torch
from pure_pytorch_3dgs import load_dataset, render, GaussianModel
from plyfile import PlyData
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ply(path):
    ply = PlyData.read(path)
    pts = ply.elements[0].data
    xyz = np.stack((pts['x'], pts['y'], pts['z']), axis=-1)
    
    model = GaussianModel(xyz.shape[0]).to(device)
    with torch.no_grad():
        model.xyz.copy_(torch.tensor(xyz, dtype=torch.float32, device=device))
        
        # Load features_dc
        f_dc = np.stack((pts['f_dc_0'], pts['f_dc_1'], pts['f_dc_2']), axis=-1)
        model.features_dc.copy_(torch.tensor(f_dc, dtype=torch.float32, device=device))
        
        # Load opacity
        op = np.array(pts['opacity'])
        model.opacity.copy_(torch.tensor(op, dtype=torch.float32, device=device).unsqueeze(1))
        
        # Load scaling
        sc = np.stack((pts['scale_0'], pts['scale_1'], pts['scale_2']), axis=-1)
        model.scaling.copy_(torch.tensor(sc, dtype=torch.float32, device=device))
        
        # Load rotation
        rot = np.stack((pts['rot_0'], pts['rot_1'], pts['rot_2'], pts['rot_3']), axis=-1)
        model.rotation.copy_(torch.tensor(rot, dtype=torch.float32, device=device))
        
    return model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

cameras = load_dataset(args.source)
model = load_ply(args.model)
bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)

print("Rendering camera 0...")
img = render(cameras[0], model, bg_color)
img_np = img.permute(1, 2, 0).detach().cpu().numpy()
img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

Image.fromarray(img_np).save(args.out)
print(f"Saved {args.out}")
