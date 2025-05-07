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
        if self.best_score is None or validation_loss < self.best_score - self.delta:
            self.best_score = validation_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
    def save_checkpoint(self, model):
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {self.path} ...")
        torch.save(model.state_dict(), self.path)
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
