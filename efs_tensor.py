import openslide
import json
import os
import numpy as np
from PIL import Image
import torch
from torchvision import models
from torchvision.transforms import ToTensor
import geopandas
from shapely.geometry import Point

classification_name = 'None'  # classification_name是全局变量，根据is_inside_annotation_center的判断给对应的patch的赋值lable

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def is_background(start_x, start_y, patch_size):
    # 定义背景颜色阈值
    background_threshold = 0.8  # 如果切片中超过80%的像素接近背景颜色，则不保存
    background_color_lower = [200, 200, 200, 0]  # 下界，包括了RGBA通道
    background_color_upper = [255, 255, 255, 255]  # 上界，包括了RGBA通道
    region = slide.read_region((start_x, start_y), 0, patch_size)  # 由于不考虑分辨率，level直接设为0
    region_array = np.array(region)
    within_background = np.all((region_array >= background_color_lower) & (region_array <= background_color_upper), axis=-1)
    background_pixels = np.sum(within_background) / np.prod(region_array.shape[:2])
    # 判断是否保存切片，如果背景像素比例低于阈值则保存
    return background_pixels > background_threshold


def is_inside_annotation_center(patch_box, geojson_path):
    global classification_name  # 使用global关键字以确保更新全局变量
    classification_name = 'Other'  # classification_name是全局变量，根据is_inside_annotation_center的判断给对应的patch的赋值lable

    # 矩形中心坐标 (x, y)
    x, y = (patch_box[0] + patch_box[2]) / 2, (patch_box[1] + patch_box[3]) / 2
    # 创建一个点对象表示矩形中心
    point = Point(x, y)
    data = geopandas.read_file(geojson_path)  # 修改一下路径

    # 然后计算每个多边形区域的面积
    data['area'] = data['geometry'].area

    for index, row in data.iterrows():
        if row['geometry'].contains(point):
            print(f"Rectangle is inside: geometry {index}，its classification is :{data['classification'][index]['name']}")
            classification_name = data['classification'][index]['name']
            return True

    return False


def extract_features_from_patch(patch, model):
    patch = patch.convert('RGB')
    patch_tensor = ToTensor()(patch).unsqueeze(0)

    with torch.no_grad():
        features = model(patch_tensor)

    return features

class_labels = {
    'basic cell': 0,
    'shadow cell': 1,
    'Other': 2,
    'None': 3
}

model = models.resnet50(pretrained=True)
model.to(device)

# 提取保存病理文件patch的特征张量
wsi = 'data/wsi'
annotation = 'data/annotation'

# 获取wsi文件夹中所有bif和svs文件的路径
wsi_files = [f for f in os.listdir(wsi) if f.endswith(('.bif', '.svs'))]

for wsi_file in wsi_files:
    wsi_path = os.path.join(wsi, wsi_file)
    annotation_file = os.path.join(annotation, os.path.splitext(wsi_file)[0] + '.geojson')
    # 打开病理图像文件
    slide = openslide.OpenSlide(wsi_path)
    # 病理文件对应的标注数据
    with open(annotation_file, 'r') as geojson_file:
        annotation_data = json.load(geojson_file)
    resolutions = [0, 1, 2, 3]

    for resolution_level in resolutions:
        patch_size = (512, 512)

        root_folder = 'datasets'
        folder_name = os.path.join(root_folder, f'window_images_resolution_{resolution_level}')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        extracted_coordinates = []

        # 遍历标注数据中的区域
        for x in range(0, slide.level_dimensions[resolution_level][0], patch_size[0]):
            for y in range(0, slide.level_dimensions[resolution_level][1], patch_size[1]):
                patch_box = (x, y, x + patch_size[0], y + patch_size[1])

                # 判断是否与先前提取的视野框重叠
                overlap = False
                for existing_box in extracted_coordinates:
                    if (patch_box[0] < existing_box[2] and patch_box[2] > existing_box[0] and
                            patch_box[1] < existing_box[3] and patch_box[3] > existing_box[1]):
                        overlap = True
                        break

                if not overlap:
                    # 在这里检查在视野框能看到的最小范围的尺度的时候，是否在标注区域内
                    # 当不是最小的尺度时，无需判断标注区域，而是根据包含的标注区域的类别标签的多少进行简单的判断
                    is_inside = False
                    if is_inside_annotation_center(patch_box, annotation_file):
                        is_inside = True
                        label = class_labels.get(classification_name, 3)  # 获取类别标签，如果匹配失败，使用3（None）作为默认值

                    if is_inside:
                        # 检查是否满足背景限制
                        if not is_background(x, y, patch_size):
                            # 在这里，将patch转换为特征张量
                            print(f"Reading patch at {x}, {y} at resolution level {resolution_level}")
                            patch = slide.read_region((x, y), resolution_level, patch_size)
                            print(f"x: {x}, y: {y}, resolution_level: {resolution_level}, patch_size: {patch_size}")

                            patch = Image.fromarray(np.array(patch))  # 转换为PIL图像
                            patch_tensor = np.array(patch)  # 将图像转换为NumPy数组
                            patch_tensor = torch.from_numpy(patch_tensor).permute(2, 0, 1)  # 调整张量形状
                            patch_tensor = patch_tensor.unsqueeze(0).float()  # 添加批次维度

                            # 使用你的特征提取模型进行特征提取
                            feature_tensor = extract_features_from_patch(patch, model)
                            feature_tensor = feature_tensor.squeeze()  # 移除批次维度

                            # 将提取的特征向量保存为.pt文件
                            matching_keys = [key for key, value in class_labels.items() if value == label]

                            if matching_keys:
                                thisclass = matching_keys[0]
                                print("对应的键为:", thisclass)
                            else:
                                print("未找到匹配的键")

                            file_name = f'{thisclass}_{label}_{os.path.splitext(wsi_file)[0]}_{x}_{y}.pt'
                            file_path = os.path.join(folder_name, file_name)
                            torch.save(feature_tensor, file_path)
                            extracted_coordinates.append(patch_box)
                            print(f"Saved patch at {x}, {y} at resolution level {resolution_level} - Label: {label}")
                        else:
                            print(f"Skipped patch at {x}, {y} at resolution level {resolution_level} - Label: {label}")
                    else:
                        print(f"Skipped patch at {x}, {y} at resolution level {resolution_level} - Label: None")

                y += patch_size[1]  # 向下移动
                x += patch_size[0]

        # 关闭病理图像文件
        slide.close()
