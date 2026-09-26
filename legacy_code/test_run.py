import sys
sys.path.append('.')
from pure_pytorch_3dgs import *

print('Testing the pure PyTorch 3DGS on tiny_dataset...')
cameras = load_dataset('C:/Users/HARSHIT/Documents/3dgs/tiny_dataset')
print(f'Loaded {len(cameras)} cameras.')

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

iterations = 5
print('Starting optimization for 5 iterations to demonstrate functionality...')
for i in range(iterations):
    optimizer.zero_grad()
    loss = 0
    for cam in cameras:
        rendered_img = render(cam, model, bg_color)
        gt_img = cam['gt_image']
        loss += torch.nn.functional.l1_loss(rendered_img, gt_img)
    loss.backward()
    optimizer.step()
    print(f'Iteration {i+1}/{iterations} - Loss: {loss.item():.4f}')
print('Training loop working correctly!')
