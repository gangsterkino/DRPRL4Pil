# Enhancing pilomatricoma diagnosis to pathologist-level accuracy using hierarchical deep learning

[![License](https://img.shields.io/github/license/用户名/仓库名)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/用户名/仓库名)](https://github.com/用户名/仓库名/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/用户名/仓库名)](https://github.com/用户名/仓库名/network)

## 目录

- [介绍](#介绍)
- [数据](#数据)
- [特征提取](#特征提取)
- [模型训练](#模型训练)
- [可视化](#可视化)
- [安装](#安装)


## 介绍

该项目旨在对毛母质瘤病理图像进行细胞级诊断。通过多尺度切片和特征提取技术，结合 ResNet50 模型进行特征张量保存，并通过多尺度迁移训练提高模型的准确性。

## 数据
使用的数据包括原始的 bif 或 kfb 病理数据，以及经过 QuPath 软件标注后保存为 GeoJSON 格式的数据，存储在 `data` 文件夹中，按照项目文件夹的指示进行组织。

## 特征提取

运行 `extract.py` 对病人病理图像数据的组织区域进行多尺度切片，并将切片图像按不同尺度保存在相应的文件夹中。使用 `extract-tensor.py` 保存通过 ResNet50 提取的特征张量。

```sh
python extract.py
```
## 模型训练
运行 train.py 进行图像训练，运行 train-tensor.py 进行张量训练，保存的模型存储在 model/model 文件夹中。
```sh
python train.py
```
## 可视化
手动锚定某一尺度的区域并在不同尺度下进行切割。使用 Grad-CAM 对切割区域进行可视化，去除边界效应，拼接小显著图，并合并指定区域的显著图以增强不同特征的表示，详见`saliency`。

## 安装

Python 3.x
TensorFlow
PyTorch
其他必要的依赖项可以在 `requirements.txt`中找到

