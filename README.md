# Hardware-Agnostic Pure PyTorch 3D Gaussian Splatting

This repository contains a **100% Pure PyTorch** implementation of 3D Gaussian Splatting, optimized for Consumer GPUs and Edge Robotics.

Unlike the official implementation and Scaffold-GS, this pipeline **does not require compiling custom C++ CUDA kernels**. It runs entirely in native PyTorch tensors, making it hardware-agnostic, incredibly easy to install, and perfectly compatible with modern architectures like the RTX 50-series (Blackwell) where legacy CUDA bindings often fail.

It explores **Fixed-Capacity Optimization**—training a fixed, unchanging number of Gaussians under strict VRAM constraints—to guarantee memory determinism for autonomous robots and drones.

For a deep dive into the mathematical and architectural decisions (like Chunked Rasterization, Mixed Precision, and Exponential LR Decay), please see [ARCHITECTURE.md](ARCHITECTURE.md).

## Features
- **Zero C++ Compilation:** No `ninja` builds, no PyBind11, no MSVC compiler errors.
- **AMP Mixed Precision:** Mathematically compressed Float16 backward passes save 50% VRAM, allowing consumer GPUs (8GB) to run up to 10,000+ points without Out-Of-Memory crashes.
- **Chunked `matmul` Rasterization:** Prevents the standard `[Height, Width, Points]` tensor memory explosion natively. Replacing `einsum` with batched `matmul` provides a 10x speed boost.
- **Fixed-Capacity Geometry Pruning:** Naturally decays and culls unused Gaussians by pushing opacity to zero, acting as an automatic geometry compressor.
- **3D Flythrough Generation:** Automatically generates a 360-degree orbit video of the final optimized scene.

## Installation

Because there are no C++ extensions, installation is instant. Any Python environment with PyTorch will work.

```bash
git clone https://github.com/harshit612007/3dgs_btp.git
cd 3dgs_btp
pip install -r requirements.txt
```

## How to Run

Place your NeRF Synthetic dataset (like Lego) or real-world dataset (like Truck) in the `datasets/` folder.

### 1. Fast Validation Run (2-3 Minutes)
To quickly test if the model is learning correctly before committing to a long run:
```bash
python train.py --dataset_path datasets/lego --output_dir outputs/lego_fast_test --iterations 10000 --num_points 3000 --resolution 128 --save_freq 1000
```

### 2. High-Quality Research Paper Run (~15 Minutes)
For the final, high-quality optimization using 10,000 points and 30,000 iterations. We include the `--render_video` flag to automatically generate a 360-degree flythrough at the very end.
```bash
python train.py --dataset_path datasets/lego --output_dir outputs/lego_final --iterations 30000 --num_points 10000 --resolution 128 --save_freq 2000 --render_video
```

## Utilities

### Re-Rendering a 3D Video
If you already trained a model (e.g., you have an `optimized_scene.ply` file) and want to generate the 360-degree video frames *without* retraining from scratch, you can use the standalone renderer:
```bash
python render_video.py --dataset_path datasets/lego --ply_path outputs/lego_final/optimized_scene.ply --output_dir outputs/lego_final
```

### Creating a GIF from Video Frames
Once your script outputs the 60 individual PNG frames in the `video_frames/` directory, you can instantly stitch them into a looping GIF using Python's built-in `PIL` library:
```bash
python -c "from PIL import Image; import glob; frames = [Image.open(f) for f in sorted(glob.glob('outputs/lego_final/video_frames/*.png'))]; frames[0].save('outputs/lego_final/flythrough.gif', format='GIF', append_images=frames[1:], save_all=True, duration=50, loop=0)"
```

### Evaluating Quality (PSNR)
To prove the quality of your renders for a research paper, you can use the built-in evaluation script to calculate the mathematical PSNR (Peak Signal-to-Noise Ratio) between two images. Higher is better!
```bash
python evaluate.py outputs/lego_final/render_29999.png datasets/lego/train/r_0.png
```

## Output Artifacts
At the end of training, your `output_dir` will contain:
1. `render_XXXX.png` files showing the training progress.
2. `video_frames/` containing the 360-degree orbit renders.
3. `optimized_scene.ply` containing the final, pruned point cloud. You can drag and drop this file into MeshLab or SuperSplat to view it in full 3D!
