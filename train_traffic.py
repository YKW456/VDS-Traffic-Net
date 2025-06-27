import copy
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os


# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Sort alphabetically for consistent class ordering
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


# Data loading and augmentation
def train_val_data_process(root_dir):
    # Validation set only gets basic processing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = CustomImageDataset(root_dir, transform=transform)

    # Split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(666)
    )

    # DataLoader
    train_loader = DataLoader(
        train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

def test_get(root_dir):
    # Test set processing (same as validation)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = CustomImageDataset(root_dir, transform=transform)

    # DataLoader
    test_loader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    return test_loader


def train_model_process(model, train_loader, val_loader, test_loader, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'test_loss': [], 'test_acc': []}

    # Create DataFrame to store correct predictions, sorted by filename
    # First generate all possible filenames from file00 to file100
    all_filenames = [f"file{i:02d}" for i in range(101)]  # file00 to file100
    correct_predictions = pd.DataFrame(index=all_filenames)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training phase (unchanged)
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double().cpu().item() / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # Validation phase (unchanged)
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        test_loss = 0.0
        test_corrects = 0

        with torch.no_grad():
            # Validation phase
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

            # Test phase and record correct predictions
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                test_corrects += torch.sum(preds == labels.data)

                # Get filenames for current batch
                batch_size = inputs.size(0)
                start_idx = i * test_loader.batch_size
                filenames = [f"file{j:02d}" for j in range(start_idx, start_idx + batch_size)]

                # Record correct predictions
                correct_mask = preds == labels.data
                for j in range(batch_size):
                    if correct_mask[j]:
                        # Record correct prediction for this epoch under the filename
                        if f"epoch_{epoch}" not in correct_predictions.columns:
                            correct_predictions[f"epoch_{epoch}"] = 0
                        correct_predictions.loc[filenames[j], f"epoch_{epoch}"] = 1

        # Calculate and record loss and accuracy (unchanged)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double().cpu().item() / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_corrects.double().cpu().item() / len(test_loader.dataset)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

        # Save best model (unchanged)
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     torch.save(model.state_dict(), './model_save/best_model_resnet34.pth')

        scheduler.step()

    # Save training history (unchanged)
    history_df = pd.DataFrame(history)
    history_df.to_csv('Output/training_history.csv', index=False)

    # Save correct predictions record (modified for clearer format)
    # Transpose DataFrame to have filenames as rows and epochs as columns
    correct_predictions = correct_predictions.fillna(0).astype(int)
    correct_predictions.to_csv('Output/correct_predictions.csv')

    return history

# Visualize training process
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    if not os.path.exists('Output'):
        os.makedirs('Output')
    plt.savefig('Output/training_curve.png')
    plt.show()


if __name__ == '__main__':
    # Initialize pretrained ResNet34
    model = models.resnet34(pretrained=True)
    num_classes = 14  # Modify according to your number of classes

    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Data paths
    data_dir = 'classification-gray'  # Replace with your data path
    test_dir = 'classification-jiguang'
    # Data loading
    train_loader, val_loader = train_val_data_process(data_dir)
    test_loader = test_get(test_dir)
    # Train model
    history = train_model_process(model, train_loader, val_loader, test_loader, num_epochs=10)

    # Visualize results
    plot_training_history(history)
