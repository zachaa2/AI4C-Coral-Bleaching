import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import pandas as pd

from dataset import CoralBleachingDataset, split_dataset, collate_fn, LABEL_MAP, LABEL_NAMES
from model import get_model, train_model

# Set the path
root_dir = '../outputs/images/cur+2.0'
metadata_file = os.path.join(root_dir, 'metadata.csv')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size according to model requirements
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization parameters for pre-trained models
                         std=[0.229, 0.224, 0.225])
])

# Split the dataset
train_metadata_file, val_metadata_file = split_dataset(metadata_file, test_size=0.2)

# Instantiate the datasets
train_dataset = CoralBleachingDataset(root_dir=root_dir, metadata_file=train_metadata_file, transform=transform)
val_dataset = CoralBleachingDataset(root_dir=root_dir, metadata_file=val_metadata_file, transform=transform)

# Use DataLoader to load data
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the number of classes
num_classes = len(LABEL_MAP)

# Load the model
model = get_model(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
num_epochs = 10

# Train the model
model, all_preds, all_labels = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
print("Confusion Matrix:")
print(cm_df)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES)
print('Classification Report:')
print(report)

