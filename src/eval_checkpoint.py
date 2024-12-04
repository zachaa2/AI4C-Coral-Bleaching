import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import CoralBleachingDataset, split_dataset
from torchvision.models import ResNet50_Weights

def load_model(checkpoint_path, num_classes, device):
    """
    Load the trained ResNet-50 model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        num_classes (int): Number of classes in the dataset.
        device (torch.device): Device to load the model on.
    
    Returns:
        model: The loaded model.
    """
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)  # Replace final layer
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the test dataset and return predictions and true labels.
    
    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for the test dataset.
        device: Device to run evaluation on.
    
    Returns:
        y_true, y_pred: True labels and predicted labels.
    """
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot and display the confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths and sources
    checkpoint_path = "resnet50_coral.pth"
    root_dir = "../data/outputs/images"
    # sources used to evaluate the checkpoint should not be the same ones it was trained on
    sources = ["CuraCao Coral Reef Assessment 2023 CUR", "Curacao Coral Reef Assessment 2023 ARU", "Altieri Biscayne Bay"]

    # Data transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained normalization
    ])

    # Load test dataset
    test_dataset = CoralBleachingDataset(root=root_dir, coralnet_sources=sources, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load trained model
    num_classes = 3  # num classes: healthy, bleached, dead
    model = load_model(checkpoint_path, num_classes, device)

    # Evaluate the model
    y_true, y_pred = evaluate_model(model, test_loader, device)

    # Print metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["healthy", "bleached", "dead"]))

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names=["healthy", "bleached", "dead"])
