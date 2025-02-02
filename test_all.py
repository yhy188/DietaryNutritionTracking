import os
import torch
from RDL_unet import UNet
from utils import keep_image_size_open
from torchvision.transforms import transforms
from torchvision.utils import save_image
import glob

# Initialize network
net = UNet().cuda()

# Load weight
weights = 'params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully loaded weights')
else:
    print('no weights file found')

# Image conversion
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Random test image folder path
random_test_image_dir = "random_test_image"

# Create a folder to save all results
root_save_path = "random_test"
if not os.path.exists(root_save_path):
    os.makedirs(root_save_path)

# Conduct 20 sets of tests
for i in range(1, 21):
    # 读取每组图像的路径
    group_dir = os.path.join(random_test_image_dir, f"group{i}")
    group_files = glob.glob(os.path.join(group_dir, '*.png'))

    # Create a save path for each set of results
    group_save_path = os.path.join(root_save_path, f"group_{i}")
    if not os.path.exists(group_save_path):
        os.makedirs(group_save_path)

    # Test the images in each group
    for file_path in group_files:
        file_name = os.path.basename(file_path)
        print(f"Group {i}, File: {file_name}")
        img = keep_image_size_open(file_path)
        img_data = transform(img).cuda()
        img_data = torch.unsqueeze(img_data, dim=0)
        out = net(img_data)
        result_path = os.path.join(group_save_path, file_name)
        save_image(out, result_path)

print("Prediction completed for all groups.")
