# Hardware-Agnostic Pure PyTorch 3D Gaussian Splatting

This repository contains a **100% Pure PyTorch** implementation of 3D Gaussian Splatting. 

Unlike the official implementation and Scaffold-GS, this pipeline **does not require compiling custom C++ CUDA kernels**. It runs entirely in native PyTorch tensors, making it hardware-agnostic, incredibly easy to install, and perfectly compatible with modern architectures like the RTX 50-series (Blackwell) where legacy CUDA bindings often fail.

It is highly optimized to run on consumer GPUs (e.g., 8GB VRAM) by utilizing chunked rasterization and memory-efficient backpropagation.

## Features
- **Zero C++ Compilation:** No `ninja` builds, no PyBind11, no MSVC compiler errors.
- **Low VRAM Mode:** Internal chunking mechanism (chunk_size=128) prevents CUDA Out-Of-Memory (OOM) errors during `.backward()`, easily fitting 256x256 resolution training inside 8GB VRAM.
- **Storage Compression:** Automatically culls invisible Gaussians (Opacity < 1%) and quantizes colors to Float16 when saving the final `.ply` file, compressing scene storage by up to 50x (e.g., 10MB -> 200KB).
- **Progress Tracking:** Periodically renders a fixed camera angle during training so you can visually watch the 3D scene materialize over time.

## Installation

Because there are no C++ extensions, installation is instant. Any Python environment with PyTorch will work.

```bash
git clone https://github.com/harshit612007/3dgs_btp.git
cd 3dgs_btp
pip install -r requirements.txt
```

## How to Run

Place your COLMAP or NeRF Synthetic dataset in the `datasets/` folder. For example, if you have a dataset named `truck`, run the following command:

```bash
python train.py --dataset_path datasets/truck --output_dir outputs/truck_run --iterations 1000 --num_points 3000 --resolution 128 --save_freq 100
```

### Arguments
* `--dataset_path`: Path to your dataset (must contain `transforms_train.json` or `transforms.json` and images).
* `--output_dir`: Where to save the progress images and the final `.ply` point cloud.
* `--iterations`: Number of training steps (default: 300).
* `--num_points`: Number of 3D Gaussians to initialize. Start with 3000-5000 for 8GB GPUs.
* `--resolution`: Resolution to train at. `128` or `256` are recommended for fast evaluation.
* `--save_freq`: How often to save a progress snapshot image (e.g. every 100 steps).

## Output
At the end of training, the script will output `optimized_scene.ply` in your specified `--output_dir`. You can open this file in MeshLab or SuperSplat to view your compressed 3D scene!
