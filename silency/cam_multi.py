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
    # weights = [0.372513, 0.367, 0.559]  # Modify weights
    weights =[0.8,0.8,0.2]

    fused_gradient = gradient_field_fusion(total_gradient_maps, weights)

    # Save the fused gradient image
    filename = os.path.basename(img_path)
    output_folder_path = os.path.join(output_folder, 'cam', 'jet/128')  # 修改保存路径
    os.makedirs(output_folder_path, exist_ok=True)

    output_path = os.path.join(output_folder_path, f'{filename[:-4]}.jpg')
    plt.imsave(output_path, fused_gradient, cmap='jet', format='jpg')  # 保存为 jpg 格式
    print(f"Fused Gradient saved: {output_path}")

    # Save the original gradient image
    output_folder_path = os.path.join(output_folder, 'cam', 'jetr/128')  # 修改保存路径
    os.makedirs(output_folder_path, exist_ok=True)

    output_path = os.path.join(output_folder_path, f'{filename[:-4]}.jpg')
    plt.imsave(output_path, grayscale_cam, cmap='jet_r', format='jpg')  # 保存为 jpg 格式
    print(f"Original Gradient saved: {output_path}")

def process_images_in_folder(input_folder, output_folder):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 3)

    model_weights_path = '/Users/tanxinyu/path2_xue/model/model/best_model_resolution_0_epoch9.pth'
    state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))

    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)
    model.load_state_dict(state_dict, strict=False)

    target_layers = [model.layer4]

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    target_categories = [0, 1, 2]  # Only 0 and 1 categories are considered

    # Process all images in the input folder
    for img_filename in os.listdir(input_folder):
        if img_filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, img_filename)
            visualize_grad_cam_single(img_path, model, target_layers, target_categories, output_folder)

def main():
    input_folder = "/Users/tanxinyu/Desktop/cuts/128"  # 修改输入文件夹的路径
    output_folder = "/Users/tanxinyu/Desktop/cuts/cam"  # 修改输出文件夹的路径

    process_images_in_folder(input_folder, output_folder)

if __name__ == '__main__':
    main()
