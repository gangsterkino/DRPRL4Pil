from cv2 import cv2
import numpy as np
import os

# 分割后的图片的文件夹，以及拼接后要保存的文件夹
pic_path = '/Users/tanxinyu/Desktop/cuts/cam/cam/jetr/512/'
pic_target = '/Users/tanxinyu/Desktop/cuts/picture/'
# 数组保存分割后图片的列数和行数，注意分割后图片的格式为x_x.jpg，x从1开始
num_width_list = []
num_lenght_list = []
# 读取文件夹下所有图片的名称
picture_names = os.listdir(pic_path)
if len(picture_names) == 0:
    print("没有文件")

else:
    # 获取分割后图片的尺寸
    img_1_1 = cv2.imread(pic_path + '1_1.jpg')
    (width, length, depth) = img_1_1.shape
    # 分割名字获得行数和列数，通过数组保存分割后图片的列数和行数
    for picture_name in picture_names:
        if picture_name.startswith('.'):
            continue
        num_width_list.append(int(picture_name.split("_")[0]))
        num_lenght_list.append(int((picture_name.split("_")[-1]).split(".")[0]))
    # 取其中的最大值
    num_width = max(num_width_list)
    num_length = max(num_lenght_list)
    # 预生成拼接后的图片
    splicing_pic = np.zeros((num_width * width, num_length * length, depth))
    # 循环复制
    for i in range(1, num_width + 1):
        for j in range(1, num_length + 1):
            img_part = cv2.imread(pic_path + '{}_{}.jpg'.format(i, j))
            splicing_pic[width * (i - 1): width * i, length * (j - 1): length * j, :] = img_part
    # 保存图片，大功告成
    cv2.imwrite(pic_target + '512_resize_jetr.jpg', splicing_pic)
    print("done!!!")

