import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from scipy import stats
from PIL import Image
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from model.train import CustomDataset, ClassificationModel, collate_fn

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据集路径和相关参数
root_folder = 'data_demo'  # 数据集根目录
batch_size = 4  # 根据你的设置调整
num_folds = 5  # k折交叉验证的折数

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 加载验证集数据
val_dataset = CustomDataset(root_folder, resolution_level=0, transform=transform)

# 定义k折交叉验证
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# 加载模型并计算准确率
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
    'Mean Accuracy': []
}

# 用于保存每个模型的准确率
model_accuracies = []

for model_path in models_paths:
    # 初始化模型
    model = ClassificationModel(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    fold_accuracies = []

    # k折交叉验证
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(val_dataset.data, val_dataset.targets)):
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                                collate_fn=collate_fn, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images, labels = batch['image'].to(device), batch['label'].to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, dim=1)
                pred_labels.extend(predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # 计算准确率
        accuracy = accuracy_score(true_labels, pred_labels)
        fold_accuracies.append(accuracy)

        print(f"Model {os.path.basename(model_path)}, Fold {fold_idx + 1} - Accuracy: {accuracy:.4f}")

    # 计算平均准确率
    mean_accuracy = np.mean(fold_accuracies)
    model_accuracies.append(fold_accuracies)

    results['Model'].append(os.path.basename(model_path))
    results['Mean Accuracy'].append(mean_accuracy)

    print(f"Model {os.path.basename(model_path)} - Mean Accuracy: {mean_accuracy:.4f}")

# 输出每个模型的平均准确率
for i, acc in enumerate(results['Mean Accuracy']):
    print(f"Model {results['Model'][i]}: Mean Accuracy = {acc:.4f}")

# 计算两两模型之间的p值
for i in range(len(model_accuracies)):
    for j in range(i+1, len(model_accuracies)):
        acc1 = model_accuracies[i]
        acc2 = model_accuracies[j]
        _, p_value = stats.ttest_rel(acc1, acc2)
        print(f"p-value between Model {os.path.basename(models_paths[i])} and Model {os.path.basename(models_paths[j])} = {p_value:.4f}")

# 输出结果
results_df = pd.DataFrame(results)
print(results_df)
