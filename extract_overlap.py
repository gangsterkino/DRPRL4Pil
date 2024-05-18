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

def is_background(start_x, start_y,level,patch_size):
    # 定义背景颜色阈值
    background_threshold = 0.8  # 如果切片中超过80%的像素接近背景颜色，则不保存
    background_color_lower = [200, 200, 200, 0]  # 下界，包括了RGBA通道
    background_color_upper = [255, 255, 255, 255]  # 上界，包括了RGBA通道
    region = slide.read_region((start_x, start_y), level, patch_size)
    region_array = np.array(region)
    within_background = np.all((region_array >= background_color_lower) & (region_array <= background_color_upper),axis = -1)
    background_pixels = np.sum(within_background) / np.prod(region_array.shape[:2])
    # 判断是否保存切片，如果背景像素比例低于阈值则保存
    return background_pixels > background_threshold


def is_inside_annotation_center(patch_box):
    global classification_name  # 使用global关键字以确保更新全局变量
    classification_name = 'Other'  # classification_name是全局变量，根据is_inside_annotation_center的判断给对应的patch的赋值lable

    # 矩形中心坐标 (x, y)
    x, y = (patch_box[0] + patch_box[2]) / 2, (patch_box[1] + patch_box[3]) / 2
    # 创建一个点对象表示矩形中心
    point = Point(x, y)
    data = geopandas.read_file('data/annotation/2301928-1_202311281318.geojson')  # 修改一下路径
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
file = 'data/wsi/2301928-1_202311281318.bif'
slide = openslide.OpenSlide(file)

# 提取文件名（不包括路径、不包括扩展名）
base_name = os.path.basename(file)  # 获取文件名，包括扩展名
filename = os.path.splitext(base_name)[0]  # 去除扩展名

# 病理文件对应的标注数据
with open('data/annotation/2301928-1_202311281318.geojson', 'r') as geojson_file:
    annotation_data = json.load(geojson_file)

resolutions = [0, 1, 2, 3]  # resolution要替换到能看全局的程度，改成动态加载相应的病理文件的分辨率的代码

# resolutions = [2, 3] # resolution要替换到能看全局的程度，改成动态加载相应的病理文件的分辨率的代码

for resolution_level in resolutions:
    scale = 1.0 / (2 ** resolution_level)
    patch_size = (512, 512)

    folder_name = f'window_images_resolution_{resolution_level}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    extracted_coordinates = []

    saved_coordinates = []  # 保存成功保存为.pt文件的坐标
    skipped_coordinates = []  # 保存跳过的坐标

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
                if resolution_level == 0 or resolution_level == 1:
                    if is_inside_annotation_center(patch_box):
                        is_inside = True
                        label = class_labels.get(classification_name, 3)  # 获取类别标签，如果匹配失败，使用3（None）作为默认值
                        print(class_labels.get(classification_name))

                    if is_inside:
                        # 检查是否满足背景限制
                        if not is_background(x, y, resolution_level, patch_size):
                            # 在这里，将patch转换为特征张量
                            print(f"Reading patch at {x}, {y} at resolution level {resolution_level}")
                            patch = slide.read_region((int(x * scale), int(y * scale)), resolution_level, patch_size)
                            print(f"x: {x}, y: {y}, scale: {scale}, resolution_level: {resolution_level}, patch_size: {patch_size}")

                            patch = Image.fromarray(np.array(patch))  # 转换为PIL图像
                            patch_tensor = np.array(patch)  # 将图像转换为NumPy数组
                            patch_tensor = torch.from_numpy(patch_tensor).permute(2, 0, 1)  # 调整张量形状
                            patch_tensor = patch_tensor.unsqueeze(0).float()  # 添加批次维度

                            # 使用你的特征提取模型进行特征提取
                            feature_tensor = extract_features_from_patch(patch, model)
                            feature_tensor = feature_tensor.squeeze()  # 移除批次维度

                            # 将提取的特征向量保存为.pt文件
                            # print(class_labels.values(label))
                            matching_keys = [key for key, value in class_labels.items() if value == label]

                            if matching_keys:
                                thisclass = matching_keys[0]
                                print("对应的键为:", thisclass)
                            else:
                                print("未找到匹配的键")

                            file_name = f'{thisclass}_{label}_{filename}_resolution_{resolution_level}_{x}_{y}.pt'
                            file_path = os.path.join(folder_name, file_name)
                            torch.save(feature_tensor, file_path)
                            extracted_coordinates.append(patch_box)
                            print(f"Saved patch at {x}, {y} at resolution level {resolution_level} - Label: {label}")
                        else:
                            skipped_coordinates.append(patch_box)
                            print(f"Skipped patch at {x}, {y} at resolution level {resolution_level} - Label: {label}")
                    else:
                        skipped_coordinates.append(patch_box)
                        print(f"Skipped patch at {x}, {y} at resolution level {resolution_level} - Label: None")
                else:
                    label = class_labels.get(classification_name, 3)
                    if not is_background(x, y, resolution_level, patch_size):
                        # 在这里，将patch转换为特征张量
                        print(f"Reading patch at {x}, {y} at resolution level {resolution_level}")
                        patch = slide.read_region((int(x * scale), int(y * scale)), resolution_level, patch_size)

                        print(
                            f"x: {x}, y: {y}, scale: {scale}, resolution_level: {resolution_level}, patch_size: {patch_size}")

                        patch = Image.fromarray(np.array(patch))  # 转换为PIL图像
                        patch_tensor = np.array(patch)  # 将图像转换为NumPy数组
                        patch_tensor = torch.from_numpy(patch_tensor).permute(2, 0, 1)  # 调整张量形状
                        patch_tensor = patch_tensor.unsqueeze(0).float()  # 添加批次维度

                        # 使用你的特征提取模型进行特征提取
                        feature_tensor = extract_features_from_patch(patch, model)
                        feature_tensor = feature_tensor.squeeze()  # 移除批次维度

                        # 将提取的特征向量保存为.pt文件
                        # print(class_labels.values(label))
                        matching_keys = [key for key, value in class_labels.items() if value == label]

                        if matching_keys:
                            thisclass = matching_keys[0]
                            print("对应的键为:", thisclass)
                        else:
                            print("未找到匹配的键")

                        file_name = f'{thisclass}_{label}_{filename}_resolution_{resolution_level}_{x}_{y}.pt'
                        # 保存对应的图片
                        # file_path = os.path.join(folder_name, file_name)
                        # torch.save(feature_tensor, file_path)
                        extracted_coordinates.append(patch_box)
                        print(f"Saved patch at {x}, {y} at resolution level {resolution_level} - Label: {label}")
                    else:
                        skipped_coordinates.append(patch_box)
                        print(f"Skipped patch at {x}, {y} at resolution level {resolution_level} - Label: {label}")

            y += patch_size[1]  # 向下移动
            x += patch_size[0]


    # 在保存特征文件后， saved_coordinates 和 skipped_coordinates 可以进一步保存到文件

# 关闭病理图像文件
slide.close()