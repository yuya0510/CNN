import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchvision import transforms
class EarlyStopping:
    def __init__(self, patience=3, delta=0, path="checkpoint.pt", verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    def __call__(self, validation_loss, model):
        # 初回または損失が改善した場合
        if self.best_score is None or validation_loss < self.best_score - self.delta:
            self.best_score = validation_loss  # ベストスコアを更新
            self.save_checkpoint(model)  # モデルを保存
            self.counter = 0  # カウンターをリセット
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
    def save_checkpoint(self, model):
        """ベストモデルを保存"""
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {self.path} ...")
        torch.save(model.state_dict(), self.path)
# データセットクラス
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB').resize((128, 128))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
            image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
# データ分割関数
def split_data_with_validation(base_folder, test_sample, val_ratio=0.2):
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    val_paths, val_labels = [], []
    label_folders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    for label, folder in enumerate(label_folders):
        folder_path = os.path.join(base_folder, folder)
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                image_files = [
                    os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path)
                    if f.endswith(('jpg', 'jpeg', 'png'))
                ]
                if subfolder == test_sample:
                    test_paths.extend(image_files)
                    test_labels.extend([label] * len(image_files))
                else:
                    split_idx = int(len(image_files) * (1 - val_ratio))
                    train_paths.extend(image_files[:split_idx])
                    train_labels.extend([label] * split_idx)
                    val_paths.extend(image_files[split_idx:])
                    val_labels.extend([label] * (len(image_files) - split_idx))
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
# CNNモデル
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
# メインスクリプト
if __name__ == "__main__":
    base_folder = '組合せ画像/100'
    label_folders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    all_samples = []
    for folder in label_folders:
        all_samples += [subfolder for subfolder in os.listdir(folder) if os.path.isdir(os.path.join(folder, subfolder))]
    overall_cm = np.zeros((3, 3))  # 混同行列の初期化
    sample_accuracies = []  # サンプルごとの精度
    for test_sample in all_samples:
        print(f"\n=== テストサンプル: {test_sample} ===")
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_data_with_validation(
            base_folder, test_sample
        )
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = CustomDataset(train_paths, train_labels, transform=transform_train)
        val_dataset = CustomDataset(val_paths, val_labels, transform=transform_test)
        test_dataset = CustomDataset(test_paths, test_labels, transform=transform_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = len(set(train_labels))
        model = CNN(num_classes=num_classes).to(device)
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        weights = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        early_stopping = EarlyStopping(patience=3, verbose=True, path=f"checkpoint_{test_sample}.pt")
        training_losses = []
        validation_losses = []
        num_epochs = 20
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            # 検証データで評価
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            validation_loss = val_loss / len(val_loader)
            training_losses.append(running_loss / len(train_loader))
            validation_losses.append(validation_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {validation_loss:.4f}")
            # Early Stoppingの適用
            early_stopping(validation_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        # ベストモデルでテストデータを評価
        model.load_state_dict(torch.load(f"checkpoint_{test_sample}.pt"))
        model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        accuracy = 100 * sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets)
        sample_accuracies.append(accuracy)
        print(f"テストデータでの分類精度: {accuracy:.2f}%")
        cm = confusion_matrix(all_targets, all_predictions, labels=range(num_classes))
        print("混同行列:")
        print(cm)
        # サンプルごとの混同行列保存
       # サンプルごとの混同行列保存
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1", "Class 2"])
        fig, ax = plt.subplots()  # 新しいプロット領域を作成
        disp.plot(cmap=plt.cm.Blues, colorbar=True, ax=ax)  # ConfusionMatrixDisplayを描画
        # ヒートマップのカラーマップ範囲を設定
        im = ax.images[0]  # ヒートマップのイメージオブジェクトを取得
        im.set_clim(0, 100)  # カラーマップの範囲を0〜100に固定
        # 目盛りを内向きに設定
        plt.tick_params(direction='in')
        # プロットを保存
        plt.savefig(f"confusion_matrix_{test_sample}.png")
        plt.close()
        # 学習曲線をプロットして保存
        plt.figure(figsize=(7, 5))
        plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss")
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
# 横軸と縦軸の固定
        plt.xticks(np.arange(0, 21, 2))  # 横軸: 0～20を2刻み
        plt.xlim(0, 20)                 # 横軸の範囲を固定
        plt.yticks(np.arange(0, 1.3, 0.2))  # 縦軸: 0～1.2を0.2刻み
        plt.ylim(0, 1.2)                # 縦軸の範囲を固定
        plt.tick_params(direction='in') # 目盛りを内向きに
        plt.tight_layout()
        plt.savefig(f"learning_curve_{test_sample}.png")
        plt.close()
    # 総合混同行列を保存
    disp = ConfusionMatrixDisplay(confusion_matrix=overall_cm, display_labels=["Class 0", "Class 1", "Class 2"])
    fig, ax = plt.subplots()  # 新しいプロット領域を作成
    disp.plot(cmap=plt.cm.Blues, colorbar=True, ax=ax)  # ConfusionMatrixDisplayを描画
    # ヒートマップのカラーマップ範囲を設定
    im = ax.images[0]  # ヒートマップのイメージオブジェクトを取得
    im.set_clim(0, 100)  # カラーマップの範囲を0〜100に固定
    # 目盛りを内向きに設定
    plt.tick_params(direction='in')
    # プロットを保存
    plt.savefig("overall_confusion_matrix.png")
    plt.close()
    print("\n=== サンプルごとの精度 ===")
    for test_sample, accuracy in zip(all_samples, sample_accuracies):
        print(f"{test_sample}: {accuracy:.2f}%")
    print(f"\n=== 平均分類精度: {np.mean(sample_accuracies):.2f}% ===")
