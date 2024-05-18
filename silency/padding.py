
import cv2
import os

# 输入图像文件路径和输出图像文件夹
input_file = '/Users/tanxinyu/Desktop/可用/new/Picture3.png'  # 替换成你的输入文件路径
output_folder = '/Users/tanxinyu/Desktop/可用/new/resize'  # 输出文件夹名称，会在当前目录下创建

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 设置resize的目标大小
target_size = (600,400)

# 获取输入文件的文件名
image_file_name = os.path.basename(input_file)

# 构建输出文件的完整路径
output_path = os.path.join(output_folder, image_file_name)

# 读取图像
image = cv2.imread(input_file)

# 进行resize
resized_image = cv2.resize(image, target_size)

# 保存resize后的图像
cv2.imwrite(output_path, resized_image)

print("图像resize完成并保存到 '{}' 文件夹中.".format(output_folder))

"""

import cv2

# 文件路径
image_path ='/Users/tanxinyu/Desktop/可用/4_new2.png'
output_path = '/Users/tanxinyu/Desktop/可用/4_new.png'

# 读取图像
image = cv2.imread(image_path)

# 切除上方和左边各20像素
cropped_image = image[90:, 30:-40]

# 将图像resize到(1000, 1000)
resized_image = cv2.resize(cropped_image, (1000, 1000))

# 保存处理后的图像
cv2.imwrite(output_path, resized_image)

print("图像处理完成并保存到 '{}'.".format(output_path))
"""
"""
import os
import cv2

# 输入文件夹路径
input_folder = '/Users/tanxinyu/Desktop/cuts/cam/cam/jetr/256_2/'
# 输出文件夹路径
output_folder = '/Users/tanxinyu/Desktop/cuts/new/256_jetr/'

# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 构建输入文件的完整路径
        input_path = os.path.join(input_folder, filename)

        # 读取图像
        image = cv2.imread(input_path)

        # 裁剪图像
        cropped_image = image[30:, 10:-50]

        # 将图像resize到(512, 512)
        resized_image = cv2.resize(cropped_image, (512, 512))

        # 构建输出文件的完整路径，保持文件名一致
        output_path = os.path.join(output_folder, filename)

        # 保存处理后的图像
        cv2.imwrite(output_path, resized_image)

        print("图像处理完成并保存到 '{}'.".format(output_path))
"""