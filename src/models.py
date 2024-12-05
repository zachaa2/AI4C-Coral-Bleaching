import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class CoralBleachingModel(nn.Module):
    def __init__(self, num_classes):
        """
        Custom model that processes images and SST values.

        Args:
            num_classes (int): Number of output classes (e.g., healthy, bleached, dead).
        """
        super(CoralBleachingModel, self).__init__()
        
        # ResNet-50 for image features
        self.image_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_image_features = self.image_model.fc.in_features
        self.image_model.fc = nn.Identity()  # Remove original FC layer

        # Fully connected layer for SST
        self.sst_fc = nn.Sequential(
            nn.Linear(1, 32),  # Input size 1 (SST value), output size 32
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Combined FC layers
        self.combined_fc = nn.Sequential(
            nn.Linear(num_image_features + 32, 128),  # Combine image and SST features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # Final output layer
        )

    def forward(self, images, sst):
        """
        Forward pass for the model.

        Args:
            images (Tensor): Batch of images.
            sst (Tensor): Batch of SST values.

        Returns:
            Tensor: Logits for each class.
        """
        # Process image through ResNet
        image_features = self.image_model(images)

        # Process SST through FC layer
        sst_features = self.sst_fc(sst)

        # Concatenate features
        combined_features = torch.cat((image_features, sst_features), dim=1)

        # Final classification
        out = self.combined_fc(combined_features)
        return out
