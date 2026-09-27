# Hardware-Agnostic 3D Gaussian Splatting (Pure PyTorch)
## Architecture & Developer Deep-Dive

This document explains the underlying math, architecture, and engineering decisions behind the custom `train.py` pipeline. It is designed to help you understand exactly what the code is doing so you can confidently modify it and write your research paper.

---

## 1. The Core Philosophy
Official 3D Gaussian Splatting (3DGS) relies on highly optimized **C++ CUDA Kernels** that dynamically spawn and delete points (Densification). While incredibly fast, these kernels:
1. Fail to compile on newer GPU architectures (like Blackwell RTX 5050).
2. Cause unpredictable memory spikes that crash low-VRAM edge devices (drones, robotics).

**Our Pipeline** replaces the CUDA kernels with a **100% Pure PyTorch** implementation. We trade extreme point counts for **infinite hardware portability** and a **fixed-capacity memory ceiling**.

### High-Level Architecture Diagram
```mermaid
graph TD
    subgraph Data Loading
        A[Dataset JSON] --> B(Images)
        A --> C(Camera Poses c2w)
    end
    
    subgraph Gaussian Initialization
        D[Random 3D Points] --> E(XYZ)
        D --> F(Features DC)
        D --> G(Scaling & Rotation)
        D --> H(Opacity)
    end
    
    subgraph Pure PyTorch Rasterization
        E --> I[World to Camera w2c]
        I --> J[Frustum Culling z > 0.1]
        G --> K[Compute 3D Covariance]
        K --> L[Project to 2D Covariance]
        J --> M[Depth Sorting]
        L --> M
        F --> M
        H --> M
        
        M --> N{Chunk-Based Alpha Compositing}
    end
    
    subgraph Fixed-Capacity Optimization
        N --> O[Rendered Image]
        B --> P[Ground Truth Image]
        O --> Q[L1 Loss + SSIM]
        P --> Q
        Q --> R[AMP Float16 Backward]
        R --> S[Adam Optimizer Step]
        S --> T[LambdaLR XYZ Decay]
        T --> E
    end
```

---

## 2. The Gaussian Model (`GaussianModel` class)
Instead of a neural network with layers, our model is an **Explicit Scene Representation**. It is simply a list of 3D points, each carrying physical properties.

We track 5 learnable parameters for `N` points:
*   **`xyz`**: The 3D position of the center of the Gaussian.
*   **`features_dc`**: The RGB color of the Gaussian. 
*   **`scaling`**: How wide/tall the Gaussian is in 3D space.
*   **`rotation`**: The 3D orientation (stored as a 4D Quaternion).
*   **`opacity`**: How solid/transparent the Gaussian is (0 to 1).

**Activation Functions:**
In PyTorch, optimizer values can swing wildly. To prevent a color from becoming "negative" or opacity becoming > 1.0, we pass the raw parameters through activation functions in the `get_()` methods:
*   `torch.sigmoid(opacity)`: Forces transparency to stay strictly between `0.0` and `1.0`.
*   `torch.exp(scaling)`: Forces the scale to always be a positive size.

---

## 3. The Rasterizer (`render()` function)
This is the heart of the engine. It takes the 3D points and flattens them into a 2D image based on a camera's perspective. 

### Step A: Projection & Culling
1.  **World-to-Camera (`w2c`)**: We multiply the `xyz` coordinates by the camera matrix to figure out where the points are relative to the camera lens.
2.  **Culling**: `valid_mask = view_pos[:, 2] > 0.1` throws away any points that are *behind* the camera. This saves massive amounts of computation.
3.  **Covariance Math**: `compute_cov3d` turns scale and rotation into a 3D blob. `project_cov3d_to_cov2d` squashes that 3D blob flat onto the 2D screen.

### Step B: Depth Sorting
To draw translucent objects correctly, you must draw them from back-to-front or front-to-back.
*   `torch.argsort(z, descending=False)`: Sorts all Gaussians by their distance (`z`) from the camera lens.

---

## 4. Chunk-Based Volume Rendering (The Memory Fix)
In a standard PyTorch matrix operation, checking every point against every pixel creates a tensor of size `(Height, Width, Num_Points)`. For a 128x128 image with 10,000 points, this intermediate tensor explodes your VRAM and crashes the GPU instantly.

**The Solution (`chunk_size = 512`):**
Instead of drawing all 10,000 points at once, we chop the sorted points into chunks of 512. 
1. We calculate the pixel overlap (`alpha`) for just 512 points.
2. We composite their colors onto the `out_color` canvas.
3. We update the global `transmittance` (how much light is blocked).
4. We throw away the memory for that chunk and move to the next 512 points.

This ensures your VRAM usage never exceeds the size of a single chunk, allowing you to run PyTorch on consumer GPUs.

---

## 5. Mixed Precision AMP (`torch.cuda.amp.autocast`)
By default, PyTorch uses **32-bit floating-point numbers (Float32)**. Every coordinate and gradient takes 4 bytes. 
By wrapping our render loop in `autocast()`, PyTorch dynamically converts the heaviest matrix multiplications into **16-bit (Float16)**.

**Why this is huge:**
*   It slashes the size of the backward-pass gradient graph by exactly **50%**.
*   It allows us to increase our point budget from 3,000 to 10,000+ points on an 8GB GPU.
*   The `GradScaler` prevents "underflow" (where tiny 16-bit gradients accidentally round down to zero) by scaling them up safely before updating the optimizer.

---

## 6. Fixed-Capacity Learning (No Densification)
Because we cannot dynamically clone/split points in PyTorch without destroying the Adam Optimizer's internal momentum states, we use a **Fixed-Capacity** approach. You start with exactly 10,000 points, and you end with exactly 10,000 points.

**The Learning Rate Schedule:**
To make this work, the model has to be carefully guided to settle into its final shape. We use a custom `LambdaLR` scheduler:
*   We rapidly decay the learning rate of the **`xyz` (Positions)** down to 1% of their original speed over 10,000 iterations.
*   We keep the learning rate for Colors and Opacities constant.
*   *Result:* The points quickly lock into their physical locations in space early in training, and spend the rest of the time purely optimizing their shapes, colors, and shadows.
