import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import argparse
from dataset import CoralBleachingDatasetWithSST, split_dataset_with_sst
from models import CoralBleachingModel


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, sst, labels in dataloader:
        images, sst, labels = images.to(device), sst.to(device), labels.to(device)

        # Forward pass
        outputs = model(images, sst)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, sst, labels in dataloader:
            images, sst, labels = images.to(device), sst.to(device), labels.to(device)

            # Forward pass
            outputs = model(images, sst)
            loss = criterion(outputs, labels)

            # Compute metrics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def save_splits(train_dataset, val_dataset, test_dataset, split_file):
    with open(split_file, 'wb') as f:
        pickle.dump((train_dataset, val_dataset, test_dataset), f)


def load_splits(split_file):
    with open(split_file, 'rb') as f:
        return pickle.load(f)


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("./images/conf_matrix_resnet_sst_cur.png")
    plt.show()
    plt.close()


def train_model(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths and sources
    root_dir = "../data/outputs/images"
    sources = ["CuraCao Coral Reef Assessment 2023 CUR"]

    # Data transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalization for ResNet-50
    ])

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset_with_sst(
        root=root_dir, coralnet_sources=sources, train_ratio=0.6, val_ratio=0.15, test_ratio=0.25, transform=data_transforms
    )

    # Save splits
    save_splits(train_dataset, val_dataset, test_dataset, "sst_dataset_splits_cur.pkl")
    print("Saved train/val/test splits...")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize custom model
    num_classes = len(train_dataset.label_map)
    model = CoralBleachingModel(num_classes=num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("Starting training...\n")
    for epoch in range(args.num_epochs):
        start_time = time.time()
            
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Epoch Duration: {epoch_duration:.2f} seconds")

    # Save the model
    torch.save(model.state_dict(), "resnet50_sst_coral_cur.pth")
    print("Model saved.")


def evaluate_model():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load saved splits
    _, _, test_dataset = load_splits("sst_dataset_splits_cur.pkl")
    print("Loaded splits successfully...")

    # Data loader for the test set
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the trained model
    num_classes = len(test_dataset.label_map)
    model = CoralBleachingModel(num_classes=num_classes)
    model.load_state_dict(torch.load("resnet50_sst_coral_cur.pth", map_location=device))
    model = model.to(device)
    print("Loaded model successfully...")

    # Evaluate the model
    print("Evaluating model...")
    model.eval()
    criterion = nn.CrossEntropyLoss()
    y_true = []
    y_pred = []

    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0

        for images, sst, labels in test_loader:
            images, sst, labels = images.to(device), sst.to(device), labels.to(device)
            outputs = model(images, sst)
            loss = criterion(outputs, labels)

            # Collect predictions and true labels for confusion matrix
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # Accumulate metrics
            running_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_loss = running_loss / total
    test_acc = correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Classification report and confusion matrix
    class_names = ["healthy", "bleached"]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    plot_confusion_matrix(y_true, y_pred, class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True, help="Mode: train or eval")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs (train mode only)")
    args = parser.parse_args()

    if args.mode == "train":
        print("Running script in train mode")
        train_model(args)
    elif args.mode == "eval":
        print("Running script in eval mode")
        evaluate_model()
