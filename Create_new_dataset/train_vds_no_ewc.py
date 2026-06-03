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
    Build pretrained ResNet34 model

    Args:
        num_classes: number of output classification categories

    Returns:
        torch.nn.Module: constructed ResNet34 network instance
    """
    # Load official ImageNet pretrained ResNet34 backbone
    model = models.resnet34(pretrained=True)

    # Replace final fully-connected layer to match target category quantity
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Sort folder name by numeric value instead of string sorting: 0,1,2,...,45
        self.classes = sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else float('inf'))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        """Traverse dataset folder to collect image path and corresponding label"""
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
        """Return total quantity of dataset samples"""
        return len(self.images)

    def __getitem__(self, idx):
        """Fetch single sample by index: read image + corresponding label"""
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def split_dataset(dataset, train_classes):
    """Split dataset into train/val/test with 9:1:1 proportion"""
    # Collect all sample indices belonging to specified training categories
    all_indices = [i for i, (_, label) in enumerate(dataset.images) if label in train_classes]
    
    # First split: 90% for train+validation set, remaining 10% for test set
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.1,  # Reserve 10% data as test set
        random_state=42,
        stratify=[dataset.images[i][1] for i in all_indices]  # Stratified sampling to keep class distribution
    )
    
    # Secondary split: divide the 90% subset into train and validation
    # Final proportion: Train=81%, Val=9%, Test=10% (overall 9:1:1)
    # test_size=0.1111 ≈ 1/9, split 10% out of 90% for validation
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.1111,
        random_state=42,
        stratify=[dataset.images[i][1] for i in train_val_indices]
    )
    
    # Construct subset dataset objects based on split indices
    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    test_data = Subset(dataset, test_indices)
    
    print(f"Dataset split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    print(f"Ratio - Train: {len(train_indices)/len(all_indices):.1%}, "
          f"Val: {len(val_indices)/len(all_indices):.1%}, "
          f"Test: {len(test_indices)/len(all_indices):.1%}")
    
    return train_data, val_data, test_data


def create_dataloaders(train_data, val_data, test_data, batch_size=32):
    """Build DataLoader for train/validation/test dataset"""
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
    """Full model training pipeline: train + validate + periodic test, save best checkpoint"""
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Lists to record loss & accuracy changes per epoch
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    test_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Train stage: enable dropout & BN training mode
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

            # Accumulate total loss and correct prediction count
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        # Update learning rate if scheduler is assigned
        if scheduler:
            scheduler.step()
            print(f"Current LR: {scheduler.get_last_lr()}")

        # Validation stage: disable gradient computation, eval mode
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

        # Calculate test set accuracy after each epoch
        test_acc = test_model(model, test_dataloader, device)
        test_acc_all.append(test_acc)

        # Calculate average loss and accuracy of current epoch
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print(f"{epoch} Train Loss: {train_loss_all[-1]:.4f} Acc: {train_acc_all[-1]:.4f}")
        print(f"{epoch} Val Loss: {val_loss_all[-1]:.4f} Acc: {val_acc_all[-1]:.4f}")
        print(f"{epoch} Test Acc: {test_acc:.4f}")

        # Update best checkpoint when validation accuracy is improved
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print(f"Time elapsed: {time_use // 60:.0f}m {time_use % 60:.0f}s")

    # Wrap training log into DataFrame for file saving
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
    """Evaluate model on test set and return overall accuracy (%)"""
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
    """Generate CSV-form detailed prediction table containing image path, true label, pred label and confidence"""
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
    
    # Fetch original file path for each test sample
    test_indices = test_dataloader.dataset.indices
    for idx in test_indices:
        image_paths.append(dataset.images[idx][0])
    
    # Assemble full prediction result table
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
    # Data augmentation & normalization pipeline consistent with ImageNet
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize full dataset from root folder
    full_dataset = CustomImageDataset(root_dir='jiguang', transform=transform)

    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Number of classes: {len(full_dataset.classes)}")
    print(f"Classes: {full_dataset.classes}")

    # Select all categories for training in this experiment
    train_classes = [i for i in range(len(full_dataset.classes))]
    print(f"Training classes: {train_classes}")

    # Execute dataset split with fixed 9:1:1 ratio
    train_data, val_data, test_data = split_dataset(full_dataset, train_classes)

    # Instantiate DataLoader, reduce batchsize to avoid out-of-memory error
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size=32
    )

    # Configure computing device: prefer CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = len(full_dataset.classes)
    model = create_resnet34_pretrained(num_classes=num_classes).to(device)
    print(f"ResNet34 pretrained model created with {num_classes} output classes")
    criterion = nn.CrossEntropyLoss()

    # Hyperparameter configuration for non-EWC training experiment
    print("=== Starting Training (No EWC) ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)

    # Launch training workflow
    train_results, best_model_wts = train_model_process(
        model, train_loader, val_loader, test_loader,
        num_epochs = 30, optimizer=optimizer, criterion=criterion,
        device=device, scheduler=scheduler
    )

    # Load optimal checkpoint for final inference
    model.load_state_dict(best_model_wts)

    # Persist best model weight and training log to disk
    torch.save(best_model_wts, "./model_save/train_vds_no_ewc_best_model.pth")
    train_results.to_csv("./model_save/train_vds_no_ewc_results.csv", index=False)

    # Generate full test-set prediction detail table
    print("=== Generating Test Predictions Table ===")
    test_predictions_table = generate_test_predictions_table(
        model, test_loader, full_dataset, device, full_dataset.classes
    )

    # Save test prediction result into CSV file
    test_predictions_table.to_csv("./model_save/test_predictions_table.csv", index=False)

    # Calculate global test-set accuracy statistics
    total_test_samples = len(test_predictions_table)
    correct_predictions = test_predictions_table['Correct'].sum()
    accuracy = correct_predictions / total_test_samples * 100

    print(f"\n=== Test Set Results Summary ===")
    print(f"Total test samples: {total_test_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Test accuracy: {accuracy:.2f}%")

    # Calculate per-category test accuracy
    print(f"\n=== Class-wise Test Accuracy ===")
    for class_idx, class_name in enumerate(full_dataset.classes):
        class_samples = test_predictions_table[test_predictions_table['True_Label_Index'] == class_idx]
        if len(class_samples) > 0:
            class_accuracy = class_samples['Correct'].sum() / len(class_samples) * 100
            print(f"Class {class_idx} ({class_name}): {class_accuracy:.2f}% ({class_samples['Correct'].sum()}/{len(class_samples)})")

    # Plot loss & accuracy training curves
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

    # Output saved file paths
    print(f"\n=== Files Saved ===")
    print(f"Model: ./model_save/train_vds_no_ewc_best_model.pth")
    print(f"Training results: ./model_save/train_vds_no_ewc_results.csv")
    print(f"Test predictions: ./model_save/test_predictions_table.csv")
    print(f"Training curves: ./model_save/training_curves.png")


if __name__ == '__main__':
    # Create output folders automatically if not exist
    os.makedirs("./model_save", exist_ok=True)
    os.makedirs("./data_save", exist_ok=True)
    main()
