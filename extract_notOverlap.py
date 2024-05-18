import concurrent.futures
import os
"""
# windows下运行openslide要添加以下文件路径
dll_directory = r' openslide的bin目录的路径'
os.add_dll_directory(dll_directory)
"""

import openslide
import numpy as np
from PIL import Image
import geopandas
from shapely.geometry import Point
import re
classification_name = 'None'

wsi_folder = 'data/wsi'
annotation_folder = 'data/annotation'


def is_background(start_x, start_y, level, patch_size, slide):
    threshold = 30  # 通道差异阈值
    background_threshold = 0.6  # 背景像素比例阈值

    region = slide.read_region((start_x, start_y), level, patch_size)
    region_array = np.array(region)

    channel_diff = np.max(region_array, axis=-1) - np.min(region_array, axis=-1)
    within_background = channel_diff <= threshold

    background_pixels = np.sum(within_background) / np.prod(region_array.shape[:2])

    return background_pixels > background_threshold


def is_background_2(image_array, threshold=20, background_threshold=0.7):
    channel_diff = np.max(image_array, axis=-1) - np.min(image_array, axis=-1)
    within_background = channel_diff <= threshold
    background_pixels = np.sum(within_background) / np.prod(image_array.shape[:2])
    return background_pixels > background_threshold


def delete_background_images(folder_path, saved_coordinates, threshold=20, background_threshold=0.7):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path).convert('RGB')
                    parts = re.split(r'[._]', file)
                    coordinates = parts[-3:-1]
                    resolution_level = parts[-4]

                    # 从文件名末尾提取 x 和 y 的坐标
                    center_x, center_y = map(int, coordinates)

                    if (center_x, center_y) not in saved_coordinates:
                        continue
                    if is_background_2(np.array(img), threshold, background_threshold):
                        os.remove(file_path)
                        # Also delete corresponding coordinates
                        print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")


def is_inside_annotation_center(patch_box, geojson_data):
    global classification_name
    classification_name = 'None'
    x, y = (patch_box[0] + patch_box[2]) / 2, (patch_box[1] + patch_box[3]) / 2
    point = Point(x, y)

    for index, row in geojson_data.iterrows():
        if row['geometry'].contains(point):
            classification_name = geojson_data['classification'][index]['name']
            return True

    return False


class_labels = {
    'basic cell': 0,
    'shadow cell': 1,
    'Other': 2,
    'None': 3
}


def save_patch_image(patch, folder_name, thisclass, label, wsi_file, resolution_level, center_x, center_y):
    file_name = f'{thisclass}_{label}_{wsi_file}_resolution_{resolution_level}_{int(center_x)}_{int(center_y)}.png'
    file_path = os.path.join(folder_name, file_name)

    with Image.fromarray(np.array(patch)).convert("RGB") as patch_image:
            patch_image.save(file_path, format="PNG")

    """
    patch_image = Image.fromarray(np.array(patch))
    patch_image = patch_image.convert("RGB")
    patch_image.save(file_path, format="PNG")
    """


def process_wsi(wsi_file):
    wsi_path = os.path.join(wsi_folder, wsi_file)
    annotation_file = os.path.join(annotation_folder, os.path.splitext(wsi_file)[0] + '.geojson')

    with openslide.OpenSlide(wsi_path) as slide:
        print(f"Opened WSI: {wsi_path}")
        with open(annotation_file, 'r') as geojson_file:
            annotation_data = geopandas.read_file(geojson_file)

        resolutions = [0, 1, 2, 3]
        patch_size = (128, 128)

        for resolution_level in resolutions:
            print(f"Processing resolution level: {resolution_level}")
            scale = 1.0 / (2 ** resolution_level)

            folder_name = f'dataset/window_images_resolution_{resolution_level}'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            extracted_coordinates = []
            skipped_coordinates = []
            saved_coordinates = []
            delete_counter = 0
            delete_threshold = 200
            edge_margin = 0

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []

                # 避免冗余计算：计算 patch_size 和 level_dimensions 一次
                level_dimensions = slide.level_dimensions[resolution_level]
                for y in range(edge_margin, level_dimensions[1] - edge_margin, patch_size[1]):
                    for x in range(edge_margin, level_dimensions[0] - edge_margin, patch_size[0]):
                        patch_box = (x, y, x + patch_size[0], y + patch_size[1])

                        center_x = (x + x + patch_size[0]) / 2
                        center_y = (y + y + patch_size[1]) / 2

                        overlap = False
                        for existing_center in extracted_coordinates:
                            existing_x, existing_y = existing_center
                            overlap_threshold = 20

                            x_offset = abs(existing_x - center_x)
                            y_offset = abs(existing_y - center_y)

                            if x_offset == overlap_threshold or y_offset == overlap_threshold:
                                overlap = True
                                break
                        # is_inside = False

                        if not overlap:
                            if not is_background(x, y, resolution_level, patch_size, slide):
                                is_inside = is_inside_annotation_center(patch_box, annotation_data)

                                if is_inside:
                                    label = class_labels.get(classification_name, 3)
                                    patch = slide.read_region((int(x * scale), int(y * scale)), resolution_level,
                                                              patch_size)
                                    matching_keys = [key for key, value in class_labels.items() if value == label]

                                    if matching_keys:
                                        thisclass = matching_keys[0]
                                        # print("对应的键为:", thisclass)
                                    else:
                                        print("未找到匹配的键")

                                    # 提交保存图像的任务到线程池
                                    futures.append(
                                        executor.submit(save_patch_image, patch, folder_name, thisclass, label,
                                                        wsi_file,
                                                        resolution_level, center_x, center_y))

                                    saved_coordinates.append((center_x, center_y))
                                    extracted_coordinates.append((center_x, center_y))
                                    delete_counter += 1

                                    if delete_counter == delete_threshold:
                                        # 等待所有保存图像的任务完成
                                        concurrent.futures.wait(futures)
                                        # 执行删除判断
                                        delete_background_images(folder_name, saved_coordinates, threshold=20,
                                                                 background_threshold=0.7)
                                        saved_coordinates = []  # 重置保存坐标列表
                                        delete_counter = 0  # 重置计数器

                                else:
                                    if resolution_level > 0:
                                        label = class_labels.get(classification_name, 3)
                                        patch = slide.read_region((int(x * scale), int(y * scale)),
                                                                  resolution_level, patch_size)
                                        thisclass = 'None'
                                        # print(f'对应的键为:{thisclass}')

                                        futures.append(
                                            executor.submit(save_patch_image, patch, folder_name, thisclass, label,
                                                            wsi_file, resolution_level, center_x, center_y))

                                        extracted_coordinates.append((center_x, center_y))
                                        delete_counter += 1

                                        if delete_counter == delete_threshold:
                                            concurrent.futures.wait(futures)
                                            delete_background_images(folder_name, saved_coordinates,
                                                                     threshold=20, background_threshold=0.7)
                                            saved_coordinates = []  # 重置保存坐标列表
                                            delete_counter = 0  # 重置计数器

                                    else:
                                        skipped_coordinates.append(patch_box)

                            else:
                                skipped_coordinates.append(patch_box)
                # 等待所有保存图像的任务完成
                concurrent.futures.wait(futures)


# 处理所有WSI文件
with concurrent.futures.ThreadPoolExecutor() as executor:
    wsi_files = [f for f in os.listdir(wsi_folder) if f.endswith(('.bif', '.svs'))]
    executor.map(process_wsi, wsi_files)
