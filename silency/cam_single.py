import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image
import cv2

def gradient_field_fusion(gradient_maps, weights):
    fused_gradient = sum(w * gm for w, gm in zip(weights, gradient_maps))
    return fused_gradient

def visualize_grad_cam_single(img_path, model, target_layers, target_categories, output_folder):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    assert os.path.exists(img_path), "File '{}' does not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    total_gradient_maps = []

    for target_category in target_categories:
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        total_gradient_maps.append(grayscale_cam)

    # Gradient field fusion
    # 0上空白变蓝色，1上空白变蓝框变红，2下变红一个度边框变蓝
    # weights = [1, 1, 0]  # Modify weights
    weights = [0.4, 0.37, 0.555]  # Modify weights

    fused_gradient = gradient_field_fusion(total_gradient_maps, weights)

    # Save the fused gradient image
    filename = os.path.basename(img_path)
    output_folder_path = os.path.join(output_folder, 'fus')
    os.makedirs(output_folder_path, exist_ok=True)

    output_path = os.path.join(output_folder_path, f'picture1_2_cam.png')
    plt.imsave(output_path, fused_gradient, cmap='jet_r')  # 使用 'viridis' 颜色映射
    print(f"Fused Gradient saved: {output_path}")

def main():
    model = models.resnet50(pretrained=False)

    model.fc = torch.nn.Linear(2048, 3)

    model_weights_path = '/Users/tanxinyu/DPRL4Pil相关/path2_xue/model/model/best_model_resolution_4_epoch7.pth'
    state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))

    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)
    model.load_state_dict(state_dict, strict=False)

    target_layers = [model.layer4]

    input_folder = "/Users/tanxinyu/Desktop/可用/new/resize"
    output_folder = "/Users/tanxinyu/Desktop/可用/new/"

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    target_categories = [0, 1, 2]  # Only 0 and 1 categories are considered

    # Process a single image in the input folder
    img_filename = "Picture1.png"  # Replace with the actual filename
    img_path = os.path.join(input_folder, img_filename)
    visualize_grad_cam_single(img_path, model, target_layers, target_categories, output_folder)

if __name__ == '__main__':
    main()
