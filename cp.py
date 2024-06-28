import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from model.train import CustomDataset, ClassificationModel, collate_fn

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
    'Cohen\'s Kappa': []
}

for model_path in models_paths:
    # 初始化模型
    model = ClassificationModel(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            images, labels = batch['image'], batch['label']
            outputs = model(images)
            _, predictions = torch.max(outputs, dim=1)
            pred_labels.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    kappa = cohen_kappa_score(true_labels, pred_labels)

    model_name = os.path.basename(model_path)
    results['Model'].append(model_name)
    results['Cohen\'s Kappa'].append(kappa)

    print(f"{model_name} - Cohen's Kappa: {kappa}")

# 输出结果
results_df = pd.DataFrame(results)
print(results_df)
