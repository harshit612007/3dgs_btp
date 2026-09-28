import argparse
import torch
import os
from train import GaussianModel, load_dataset, render_trajectory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    print(f"Loading dataset from {args.dataset_path}...")
    cameras = load_dataset(args.dataset_path, 128, False)
    
    print(f"Loading trained point cloud from {args.ply_path}...")
    model = GaussianModel(0) # Initialize empty
    model.load_ply(args.ply_path)
    model.to(device)
    
    render_trajectory(model, cameras, args.output_dir, num_frames=60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
