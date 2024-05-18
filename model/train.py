import argparse
import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import logging
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from torchvision.transforms.functional import resize


from torchvision.datasets import ImageFolder

class CustomDataset(Dataset):
    def __init__(self, root_folder, resolution_level, transform=None):
        self.root_folder = root_folder
        self.resolution_level = resolution_level
        self.transform = transform

        folder_name = f'window_images_resolution_{resolution_level}'
        folder_path = os.path.join(root_folder, folder_name)

        # 获取文件夹下所有图片的路径
        self.image_files = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
        self.class_to_idx = {'basic cell': 0, 'shadow cell': 1, 'Other': 2}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')  # 读取图像

        # Extract information from the image file name
        file_name = os.path.basename(image_path)
        parts = file_name.split('_')
        class_name = parts[0]  # 类别信息在文件名的第一个部分

        # 获取类别索引
        label = self.class_to_idx.get(class_name, -1)
        if label == -1:
            # 跳过该样本
            return None

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}


def collate_fn(batch):
    images = []
    labels = []

    for item in batch:
        if item is not None and item['label'] is not None and item['label'] != 3:
            images.append(item['image'])
            labels.append(item['label'])

    if len(images) == 0 or len(labels) == 0:
        # 处理所有项都是None的情况
        return None

    # 调整图像大小为相同的尺寸
    target_size = (512, 512)
    images = [resize(img, target_size) for img in images]

    images = [transforms.ToTensor()(img).to(device) for img in images]

    return {'image': torch.stack(images), 'label': torch.tensor(labels).to(device)}

class ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


def get_args():
    #  python mdoel/new.py --root data_demo
    parser = argparse.ArgumentParser(description='Train the ResNet model for image classification')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate',
                        dest='lr')
    parser.add_argument('--val_percent', '-vp', metavar='VP', type=float, default=10,
                        help='Percentage of data for validation')
    parser.add_argument('--save_checkpoint', action='store_true', default=True, help='Save model checkpoint')
    parser.add_argument('--root', type=str, required=True, default='data_demo',
                        help='Root directory of the classification dataset')

    return parser.parse_args()



def train_classification_model(
        model,
        device,
        train_loader,
        val_loader,
        resolution_level,
        epochs: int = 5,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        results_csv: str = 'results/results.csv',
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    all_results = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        true_labels_train = []
        outputs_train = []
        predicted_labels_train = []

        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='image') as pbar:
            for batch in train_loader:
                features, labels = batch['image'].to(device), batch['label'].to(device)

                # 处理标签为'None'的情况
                none_mask = labels == class_labels['None']
                none_samples = features[none_mask]
                none_labels = labels[none_mask]

                if len(none_samples) > 0:
                    with torch.no_grad():
                        # Forward pass to get predicted labels for 'None' samples
                        predicted_labels = torch.argmax(model(none_samples), dim=1)
                    none_labels = predicted_labels

                # 合并已处理的'None'样本和其他样本
                features = torch.cat([features[~none_mask], none_samples], dim=0)
                labels = torch.cat([labels[~none_mask], none_labels], dim=0)

                optimizer.zero_grad()

                outputs = model(features)
                predicted_labels = torch.argmax(outputs, dim=1)

                # 在计算 softmax_outputs 时添加 detach()
                softmax_outputs = F.softmax(outputs, dim=1).detach()


                # 使用 softmax_outputs.cpu().numpy()
                outputs_train.extend(softmax_outputs.cpu().numpy())
                predicted_labels_train.extend(predicted_labels.cpu().numpy())
                supervised_loss = criterion(outputs, labels)
                loss = supervised_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                true_labels_train.extend(labels.cpu().numpy())

                accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())


                precision = precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted',
                                            zero_division=0)
                recall = recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted',
                                      zero_division=0)
                f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted', zero_division=0)

                pbar.set_postfix(loss=loss.item(), accuracy=accuracy, precision=precision, recall=recall, f1=f1)
                pbar.update(features.size(0))

        accuracy_train = correct_predictions / total_samples
        precision_train = precision
        recall_train = recall
        f1_train = f1

        logging.info(f'Epoch {epoch + 1}/{epochs}, Training Accuracy: {accuracy_train}, '
                     f'Precision: {precision_train}, Recall: {recall_train}, F1: {f1_train}')

        # 在 model.eval() 后的代码中
        model.eval()
        true_labels_val = []
        predicted_labels_val = []
        outputs_val = []

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch['image'].to(device), batch['label'].to(device)

                # 处理标签为'None'的情况
                none_mask = labels == class_labels['None']
                none_samples = inputs[none_mask]
                none_labels = labels[none_mask]

                if len(none_samples) > 0:
                    with torch.no_grad():
                        # Forward pass to get predicted labels for 'None' samples
                        predicted_labels = torch.argmax(model(none_samples), dim=1)
                    none_labels = predicted_labels

                # 合并已处理的'None'样本和其他样本
                inputs = torch.cat([inputs[~none_mask], none_samples], dim=0)
                labels = torch.cat([labels[~none_mask], none_labels], dim=0)

                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1).detach()  # 使用 Softmax 转换为概率

                _, predicted = torch.max(outputs.data, 1)

                true_labels_val.extend(labels.cpu().numpy())
                predicted_labels_val.extend(predicted.cpu().numpy())
                outputs_val.extend(probabilities.cpu().numpy())  # 使用概率而不是原始输出

        acc_val = accuracy_score(true_labels_val, predicted_labels_val)
        precision_val = precision_score(true_labels_val, predicted_labels_val, average='weighted', zero_division=0)
        recall_val = recall_score(true_labels_val, predicted_labels_val, average='weighted', zero_division=0)
        f1_val = f1_score(true_labels_val, predicted_labels_val, average='weighted', zero_division=0)

        logging.info(f'Epoch {epoch + 1}/{epochs}, Validation Accuracy: {acc_val}, '
                     f'Precision: {precision_val}, Recall: {recall_val}, F1: {f1_val}')

        if save_checkpoint and acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(model.state_dict(), f'model/best_model_resolution_{resolution_level}_epoch{epoch + 1}.pth')

        confusion_mat_train = confusion_matrix(true_labels_train, predicted_labels_train)
        roc_auc_train = roc_auc_score(true_labels_train, outputs_train, multi_class='ovr', average='weighted')

        confusion_mat_val = confusion_matrix(true_labels_val, predicted_labels_val)
        roc_auc_val = roc_auc_score(true_labels_val, outputs_val, multi_class='ovr', average='weighted')

        results = {
                'epoch': epoch + 1,
                'train_accuracy': accuracy_train,
                'train_precision': precision_train,
                'train_recall': recall_train,
                'train_f1': f1_train,
                'val_accuracy': acc_val,
                'val_precision': precision_val,
                'val_recall': recall_val,
                'val_f1': f1_val,
                'confusion_matrix_train': confusion_mat_train,
                'roc_auc_train': roc_auc_train,
                'confusion_matrix_val': confusion_mat_val,
                'roc_auc_val': roc_auc_val,
            }

        all_results.append(results)

        results_df = pd.DataFrame(all_results)
        print(f"Writing results to: {results_csv}")
        print(f"Results DataFrame:\n{results_df}")
        results_df.to_csv(results_csv, index=False)
        print("CSV write successful!")


if __name__ == '__main__':
    class_labels = {'basic cell': 0, 'shadow cell': 1, 'Other': 2, 'None': 3}
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')
    num_classes = 3
    resolutions = [4]

    for resolution_level in resolutions:
        print(f"start train resolution{resolution_level}")
        csv_path = f'logs/resolution_{resolution_level}.csv'
        dataset = CustomDataset(args.root, resolution_level=resolution_level, transform=None)

        n_val = int(len(dataset) * (args.val_percent / 100))
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

        loader_args = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_fn)
        train_loader = DataLoader(train_set, shuffle=False, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


        # Initialize the model for resolution_level == 0
        classification_model = ClassificationModel(num_classes=num_classes)
        classification_model.to(device)
        """
        if resolution_level > 0:
            previous_model_path = f'model/best_model_resolution_{resolution_level - 1}_epoch{args.epochs}.pth'
            print(f"loading previous model  {previous_model_path}")
            classification_model.load_state_dict(torch.load(previous_model_path))
       """


        try:
            train_classification_model(
                model=classification_model,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                resolution_level=resolution_level,
                epochs=args.epochs,
                learning_rate=args.lr,
                save_checkpoint=args.save_checkpoint,
                results_csv=csv_path,
            )

            torch.save(classification_model.state_dict(), f'model/best_model_resolution_{resolution_level}_epoch{args.epochs}.pth')


        except Exception as e:
            if "MemoryError" in str(e):
                logging.error('Detected MemoryError! Handle as needed.')
                torch.cuda.empty_cache()
            else:
                raise e
        except Exception as e:
            logging.error(f'An unexpected error occurred: {str(e)}')
