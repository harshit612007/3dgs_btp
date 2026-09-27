import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math

def calculate_psnr(img1_path, img2_path):
    # Load images and convert to tensors
    img1 = torch.tensor(np.array(Image.open(img1_path).convert("RGB"))).float() / 255.0
    img2 = torch.tensor(np.array(Image.open(img2_path).convert("RGB"))).float() / 255.0
    
    # Resize img2 to match img1 if they are different resolutions
    if img1.shape != img2.shape:
        img2 = F.interpolate(img2.permute(2, 0, 1).unsqueeze(0), size=(img1.shape[0], img1.shape[1]), mode='bilinear').squeeze(0).permute(1, 2, 0)
        
    # Calculate Mean Squared Error (MSE)
    mse = F.mse_loss(img1, img2)
    
    if mse == 0:
        return float('inf')
        
    # Calculate PSNR
    psnr = -10. * torch.log10(mse)
    return psnr.item()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <image1.png> <image2.png>")
        sys.exit(1)
        
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    try:
        psnr = calculate_psnr(img1_path, img2_path)
        print(f"\n[Quality Comparison]")
        print(f"Image 1: {img1_path}")
        print(f"Image 2: {img2_path}")
        print(f"PSNR Score: {psnr:.2f} dB (Higher is better!)\n")
    except Exception as e:
        print(f"Error comparing images: {e}")
