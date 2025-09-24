import copy
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch.optim.lr_scheduler import MultiStepLR
import os
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import numpy as np
from torch.functional import F
from sklearn.model_selection import train_test_split
from model_net import *


def create_resnet34_pretrained(num_classes):
    """
    创建ResNet34预训练模型

    Args:
        num_classes: 输出类别数

    Returns:
        torch.nn.Module: ResNet34模型
    """
    # 加载预训练的ResNet34模型
    model = models.resnet34(pretrained=True)

    # 修改最后一层以适应我们的类别数
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Sort by numeric value instead of string to get correct order: 0,1,2,...,45
        self.classes = sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else float('inf'))
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


def split_dataset(dataset, train_classes):
    """Split dataset based on classes with 9:1:1 ratio (train:val:test)"""
    # Get all indices for the specified classes
    all_indices = [i for i, (_, label) in enumerate(dataset.images) if label in train_classes]
    
    # First split: 90% for train+val, 10% for test
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.1,  # 10% for test
        random_state=42,
        stratify=[dataset.images[i][1] for i in all_indices]  # Stratify by class
    )
    
    # Second split: from the 90%, split into 90% train and 10% val
    # This gives us final ratio of 81% train, 9% val, 10% test
    # To get exactly 9:1:1, we need to adjust the split
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.1111,  # 10/90 = 0.1111 to get 9:1 ratio from the remaining 90%
        random_state=42,
        stratify=[dataset.images[i][1] for i in train_val_indices]
    )
    
    # Create subsets
    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    test_data = Subset(dataset, test_indices)
    
    print(f"Dataset split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    print(f"Ratio - Train: {len(train_indices)/len(all_indices):.1%}, "
          f"Val: {len(val_indices)/len(all_indices):.1%}, "
          f"Test: {len(test_indices)/len(all_indices):.1%}")
    
    return train_data, val_data, test_data


def create_dataloaders(train_data, val_data, test_data, batch_size=32):
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
                        num_epochs, optimizer, criterion, device, scheduler=None):
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


def generate_test_predictions_table(model, test_dataloader, dataset, device, class_names):
    """Generate detailed predictions table for test set"""
    model.eval()
    
    predictions = []
    true_labels = []
    image_paths = []
    confidences = []
    
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = model(b_x)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(b_y.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())
    
    # Get image paths for test samples
    test_indices = test_dataloader.dataset.indices
    for idx in test_indices:
        image_paths.append(dataset.images[idx][0])
    
    # Create detailed results table
    results_table = pd.DataFrame({
        'Image_Path': image_paths,
        'True_Label_Index': true_labels,
        'True_Label_Name': [class_names[label] for label in true_labels],
        'Predicted_Label_Index': predictions,
        'Predicted_Label_Name': [class_names[pred] for pred in predictions],
        'Confidence': confidences,
        'Correct': [true == pred for true, pred in zip(true_labels, predictions)]
    })
    
    return results_table


def main():
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = CustomImageDataset(root_dir='jiguang', transform=transform)

    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Number of classes: {len(full_dataset.classes)}")
    print(f"Classes: {full_dataset.classes}")

    # Define training classes (all classes for first phase training)
    train_classes = [i for i in range(len(full_dataset.classes))]  # All classes
    print(f"Training classes: {train_classes}")

    # Split dataset with 9:1:1 ratio
    train_data, val_data, test_data = split_dataset(full_dataset, train_classes)

    # Create dataloaders with smaller batch size to avoid memory issues
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size=32  # Reduced from 128 to 32
    )

    # Initialize model and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = len(full_dataset.classes)
    model = create_resnet34_pretrained(num_classes=num_classes).to(device)
    print(f"ResNet34 pretrained model created with {num_classes} output classes")
    criterion = nn.CrossEntropyLoss()

    # Training parameters (same as train_vds.py)
    print("=== Starting Training (No EWC) ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)

    # Train the model
    train_results, best_model_wts = train_model_process(
        model, train_loader, val_loader, test_loader,
        num_epochs = 30, optimizer=optimizer, criterion=criterion,
        device=device, scheduler=scheduler
    )

    # Load best model for final evaluation
    model.load_state_dict(best_model_wts)

    # Save the best model
    torch.save(best_model_wts, "./model_save/train_vds_no_ewc_best_model.pth")
    train_results.to_csv("./model_save/train_vds_no_ewc_results.csv", index=False)

    # Generate detailed test predictions table
    print("=== Generating Test Predictions Table ===")
    test_predictions_table = generate_test_predictions_table(
        model, test_loader, full_dataset, device, full_dataset.classes
    )

    # Save test predictions table
    test_predictions_table.to_csv("./model_save/test_predictions_table.csv", index=False)

    # Print summary statistics
    total_test_samples = len(test_predictions_table)
    correct_predictions = test_predictions_table['Correct'].sum()
    accuracy = correct_predictions / total_test_samples * 100

    print(f"\n=== Test Set Results Summary ===")
    print(f"Total test samples: {total_test_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Test accuracy: {accuracy:.2f}%")

    # Class-wise accuracy
    print(f"\n=== Class-wise Test Accuracy ===")
    for class_idx, class_name in enumerate(full_dataset.classes):
        class_samples = test_predictions_table[test_predictions_table['True_Label_Index'] == class_idx]
        if len(class_samples) > 0:
            class_accuracy = class_samples['Correct'].sum() / len(class_samples) * 100
            print(f"Class {class_idx} ({class_name}): {class_accuracy:.2f}% ({class_samples['Correct'].sum()}/{len(class_samples)})")

    # Plot training results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_results['epoch'], train_results['train_loss'], 'r-', label='Train Loss')
    plt.plot(train_results['epoch'], train_results['val_loss'], 'b-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_results['epoch'], train_results['train_acc'], 'r-', label='Train Acc')
    plt.plot(train_results['epoch'], train_results['val_acc'], 'b-', label='Val Acc')
    plt.plot(train_results['epoch'], train_results['test_acc']/100, 'g-', label='Test Acc')
    plt.title('Training, Validation and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./model_save/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n=== Files Saved ===")
    print(f"Model: ./model_save/train_vds_no_ewc_best_model.pth")
    print(f"Training results: ./model_save/train_vds_no_ewc_results.csv")
    print(f"Test predictions: ./model_save/test_predictions_table.csv")
    print(f"Training curves: ./model_save/training_curves.png")


if __name__ == '__main__':
    os.makedirs("./model_save", exist_ok=True)
    os.makedirs("./data_save", exist_ok=True)
    main()
