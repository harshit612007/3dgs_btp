# This file is just to verify that the latest git pushes are successfully reaching your Molab instance!
# You should see this file appear after you run 'git pull' on the cluster.

# - Reverted to Hardware-Agnostic Dense Math (No Triton / No torch.compile)
# - Hardcoded chunk_size to 1024 to maximize unthrottled memory bandwidth on 96GB A100 Molab nodes
# - Completely stripped Checkpointing to guarantee raw execution speed
# - Added tum2nerf.py pipeline to parse and mathematically align TUM RGB-D optical tracking datasets to native NeRF coordinate space