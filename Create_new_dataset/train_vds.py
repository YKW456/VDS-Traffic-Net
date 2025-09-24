import copy
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch.optim.lr_scheduler import MultiStepLR
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import numpy as np
from torch.functional import F
from sklearn.model_selection import train_test_split
from model_net import *

# EWC implementation class
class EWC:
    def __init__(self, model, dataloader, device, importance=100):
        self.model = model
        self.importance = importance
        self.device = device
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p.data)

        self.model.train()
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = F.log_softmax(self.model(data), dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    fisher[n] += (p.grad ** 2) / len(dataloader)

        return fisher

    def penalty(self, new_model):
        loss = 0
        for n, p in new_model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return loss * (self.importance / 2)


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Sort to ensure consistent class ordering
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path):
                    images.append((img_path, self.class_to_idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def split_dataset(dataset, train_classes,train_classes2):
    """Split dataset based on classes"""
    train_indices = [i for i, (_, label) in enumerate(dataset.images) if label in train_classes]
    train_indices_extra = [i for i, (_, label) in enumerate(dataset.images) if label in train_classes2]

    train_indices, split_indices = train_test_split(
        train_indices,
        test_size=0.2,
        random_state=42,
    )
    train_indices_extra, split_indices2 = train_test_split(
        train_indices_extra,
        test_size=0.2,
        random_state=42,
    )
    val_indices, test_indices = train_test_split(
        split_indices,
        test_size=0.5,
        random_state=42,
    )
    val_indices_extra, test_indices_extra = train_test_split(
        split_indices2,
        test_size=0.5,
        random_state=42,
    )
    train_indices2 = train_indices + train_indices_extra
    val_indices2 = val_indices + val_indices_extra
    test_indices2 = test_indices + test_indices_extra
    # Create subsets
    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)

    train_data2 = Subset(dataset, train_indices2)
    val_data2 = Subset(dataset, val_indices2)
    test_data2 = Subset(dataset, test_indices2)
    return train_data, val_data, train_data2, val_data2, test_data2


def create_dataloaders(train_data, val_data, test_data, batch_size=128):
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    return train_dataloader, val_dataloader, test_dataloader


def train_model_process(model, train_dataloader, val_dataloader, test_dataloader,
                        num_epochs, optimizer, criterion, device,
                        ewc=None, ewc_lambda=3.0, scheduler=None):
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    test_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Training phase
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_num = 0

        for b_x, b_y in train_dataloader:
            b_x, b_y = b_x.to(device), b_y.to(device)

            optimizer.zero_grad()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            # Add EWC penalty
            if ewc is not None:
                ewc_loss = ewc.penalty(model)
                loss += ewc_lambda * ewc_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        if scheduler:
            scheduler.step()
            print(f"Current LR: {scheduler.get_last_lr()}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        with torch.no_grad():
            for b_x, b_y in val_dataloader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                output = model(b_x)
                pre_lab = torch.argmax(output, dim=1)
                loss = criterion(output, b_y)

                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)

        # Testing phase
        test_acc = test_model(model, test_dataloader, device)
        test_acc_all.append(test_acc)

        # Calculate metrics
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print(f"{epoch} Train Loss: {train_loss_all[-1]:.4f} Acc: {train_acc_all[-1]:.4f}")
        print(f"{epoch} Val Loss: {val_loss_all[-1]:.4f} Acc: {val_acc_all[-1]:.4f}")
        print(f"{epoch} Test Acc: {test_acc:.4f}")

        # Save best model
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print(f"Time elapsed: {time_use // 60:.0f}m {time_use % 60:.0f}s")

    # Save results
    train_process = pd.DataFrame({
        "epoch": range(num_epochs),
        "train_loss": train_loss_all,
        "val_loss": val_loss_all,
        "train_acc": train_acc_all,
        "val_acc": val_acc_all,
        "test_acc": test_acc_all
    })

    return train_process, best_model_wts


def test_model(model, test_dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = model(b_x)
            _, predicted = torch.max(output.data, 1)
            total += b_y.size(0)
            correct += (predicted == b_y).sum().item()
    return 100 * correct / total


def main():
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = CustomImageDataset(root_dir='jiguang', transform=transform)

    # Define class splits
    # Phase 1 training classes (1-6)
    phase1_train_classes = [i for i in range(0,46)]

    # phase1_train_classes = [0, 1, 2, 3, 4, 5]  # Note: Python uses 0-based indexing

    # Phase 2 training classes (7-8)
    phase2_train_classes = []


    # Split dataset
    phase1_train, phase1_val, phase2_train, phase2_val,test_data = split_dataset(full_dataset
     ,phase1_train_classes,phase2_train_classes)


    # Create dataloaders
    phase1_train_loader, phase1_val_loader, test_loader = create_dataloaders(
        phase1_train, phase1_val, test_data
    )
    #
    # # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet2_simple(num_classes=8).to(device)  # Create model instance first

    criterion = nn.CrossEntropyLoss()
    # Phase 1 training (Classes 1-6)
    print("=== Phase 1 Training (Classes 1-6) ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    scheduler = MultiStepLR(optimizer, milestones=[15, 35], gamma=0.5)


    phase1_results, phase1_best_wts = train_model_process(
        model, phase1_train_loader, phase1_val_loader, test_loader,
        num_epochs=3, optimizer=optimizer, criterion=criterion,
        device=device, scheduler=scheduler
    )

    # Save phase 1 model
    torch.save(phase1_best_wts, "./model_save/phase1_best_model.pth")
    phase1_results.to_csv("./data_save/phase1_results.csv", index=False)
    # model.load_state_dict(torch.load("./model_save/phase1_best_model.pth"))  # Then load state dict
    # Compute EWC
    # print("Computing EWC...")
    # ewc = EWC(model, phase1_train_loader, device, importance=1)
    #
    # # Phase 2 training (Classes 7-8)
    # print("=== Phase 2 Training (Classes 7-8) ===")
    # # Re-split dataset
    #
    #
    # phase2_train_loader, phase2_val_loader, _ = create_dataloaders(
    #     phase2_train, phase2_val, test_data
    # )
    #
    # # Reinitialize optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    # scheduler = MultiStepLR(optimizer, milestones=[20, 30], gamma=0.5)
    #
    # phase2_results, phase2_best_wts = train_model_process(
    #     model, phase2_train_loader, phase2_val_loader, test_loader,
    #     num_epochs=100, optimizer=optimizer, criterion=criterion,
    #     device=device, ewc=ewc, ewc_lambda = 1, scheduler=scheduler
    # )
    #
    # # Save final model
    # torch.save(phase2_best_wts, "./model_save/final_model.pth")
    # phase2_results.to_csv("./data_save/phase2_results.csv", index=False)

    # Plot results
    plt.figure(figsize=(15, 5))

    # Phase 1 results
    plt.subplot(1, 2, 1)
    plt.plot(phase1_results['epoch'], phase1_results['train_acc'], 'r-', label='Phase1 Train Acc')
    plt.plot(phase1_results['epoch'], phase1_results['val_acc'], 'b-', label='Phase1 Val Acc')
    plt.plot(phase1_results['epoch'], phase1_results['test_acc']/100, 'g-', label='Phase1 Test Acc')
    plt.title('Phase 1 Training (Classes 1-6)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # # Phase 2 results
    # plt.subplot(1, 2, 2)
    # plt.plot(phase2_results['epoch'], phase2_results['train_acc'], 'r-', label='Phase2 Train Acc')
    # plt.plot(phase2_results['epoch'], phase2_results['val_acc'], 'b-', label='Phase2 Val Acc')
    # plt.plot(phase2_results['epoch'], phase2_results['test_acc']/100, 'g-', label='Phase2 Test Acc')
    # plt.title('Phase 2 Training (Classes 7-8) with EWC')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    os.makedirs("./model_save", exist_ok=True)
    os.makedirs("./data_save", exist_ok=True)
    main()