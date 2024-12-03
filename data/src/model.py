import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights
from tqdm import tqdm

def get_model(num_classes):
    # Load the pre-trained DenseNet model
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    # Modify the classifier to match the number of classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model = model.to(device)
    all_preds = []
    all_labels = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = batch['images'].to(device)
            labels = batch['labels'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0

        # Lists to collect predictions and labels per epoch
        all_preds_epoch = []
        all_labels_epoch = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                inputs = batch['images'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total_samples += inputs.size(0)

                # Collect predictions and labels
                all_preds_epoch.extend(preds.cpu().numpy())
                all_labels_epoch.extend(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects.double() / val_total_samples

        print(f'Validation - Loss: {val_epoch_loss:.4f} - Acc: {val_epoch_acc:.4f}')

        # Append to overall lists
        all_preds.extend(all_preds_epoch)
        all_labels.extend(all_labels_epoch)

    print('Training complete')
    return model, all_preds, all_labels