import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM  # Assuming you have a utility file with GradCAM implementation
import cv2

def gradient_field_fusion(gradient_maps, weights):
    fused_gradient = sum(w * gm for w, gm in zip(weights, gradient_maps))
    return fused_gradient

def calculate_average_sis(model, dataset_folder, target_layers, target_categories, output_folder, p=10):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize list to store SIS scores
    sis_scores = []

    # Process each image in the dataset folder
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(dataset_folder, filename)

            assert os.path.exists(img_path), f"File '{img_path}' does not exist."

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
            weights = [0.4, 0.37, 0.555]  # Modify weights according to your fusion method
            fused_gradient = gradient_field_fusion(total_gradient_maps, weights)

            # Calculate SIS for this image
            top_p_percent_indices = get_top_p_percent_indices(fused_gradient, p)
            image_with_deleted_pixels = delete_top_p_pixels(img, top_p_percent_indices)

            score_original = model(torch.unsqueeze(img_tensor, dim=0))[0]
            score_delete = model(torch.unsqueeze(data_transform(image_with_deleted_pixels), dim=0))[0]

            sis_score = score_original - score_delete
            sis_scores.append(sis_score.item())  # Assuming score is a scalar tensor

            # Save the fused gradient image
            output_path = os.path.join(output_folder, f'{filename}_cam.png')
            plt.imsave(output_path, fused_gradient, cmap='jet_r')  # Save fused gradient image
            print(f"Fused Gradient saved: {output_path}")

    # Calculate average SIS across all images
    average_sis = np.mean(sis_scores)
    return average_sis

# Helper function to get top p% indices
def get_top_p_percent_indices(saliency_map, p):
    threshold = np.percentile(saliency_map, 100 - p)
    return np.where(saliency_map >= threshold)

# Helper function to delete top p% pixels
def delete_top_p_pixels(image, indices):
    image_with_deleted_pixels = np.copy(image)
    image_with_deleted_pixels[indices] = 0  # Setting high saliency pixels to zero
    return image_with_deleted_pixels

def main():
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 3)  # Adjust based on your model output

    model_weights_path = '/path/to/your/model_weights.pth'
    state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Ensure output folder exists
    output_folder = "/path/to/your/output/folder"
    os.makedirs(output_folder, exist_ok=True)

    # Load model weights
    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)
    model.load_state_dict(state_dict, strict=False)

    target_layers = [model.layer4]  # Adjust based on your model architecture

    dataset_folder = "/path/to/your/dataset/folder"
    target_categories = [0, 1, 2]  # Adjust based on your target categories

    # Calculate average SIS for the dataset
    average_sis = calculate_average_sis(model, dataset_folder, target_layers, target_categories, output_folder)

    print(f'Average SIS for the dataset: {average_sis}')

if __name__ == '__main__':
    main()
