import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from model.train import CustomDataset, ClassificationModel,collate_fn
# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据集路径和相关参数
root_folder = 'data_demo'  # 数据集根目录
batch_size = 4  # 根据你的设置调整

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 直接加载验证集
val_dataset = CustomDataset(root_folder, resolution_level=0, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

# 定义计算 Top-k 一致性的函数
def top_k_accuracy(y_true, y_pred_probs, k=1):
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] in top_k_preds[i]:
            correct += 1
    return correct / len(y_true)

# 加载模型并计算一致性
models_paths = [
    'model/best_model_resolution_0_epoch10.pth',
    'model/best_model_resolution_1_epoch10.pth',
    'model/best_model_resolution_2_epoch10.pth',
    'model/best_model_resolution_3_epoch10.pth',
    'model/best_model_resolution_4_epoch10.pth',
    'model/best_model_resolution_5_epoch10.pth'
]

# 记录结果
results = {
    'Model': [],
    'Top-1 Accuracy': [],
    'Top-2 Accuracy': [],
    'Top-3 Accuracy': []
}

for model_path in models_paths:
    # 初始化模型
    model = ClassificationModel(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    true_labels = []
    pred_probs = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            images, labels = batch['image'], batch['label']
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            pred_probs.extend(probabilities.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)

    top_1_acc = top_k_accuracy(true_labels, pred_probs, k=1)
    top_2_acc = top_k_accuracy(true_labels, pred_probs, k=2)
    top_3_acc = top_k_accuracy(true_labels, pred_probs, k=3)

    model_name = os.path.basename(model_path)
    results['Model'].append(model_name)
    results['Top-1 Accuracy'].append(top_1_acc)
    results['Top-2 Accuracy'].append(top_2_acc)
    results['Top-3 Accuracy'].append(top_3_acc)

    print(f"{model_name} - Top-1 Accuracy: {top_1_acc}, Top-2 Accuracy: {top_2_acc}, Top-3 Accuracy: {top_3_acc}")

# 输出结果
results_df = pd.DataFrame(results)
print(results_df)
