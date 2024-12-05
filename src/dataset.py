import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class CoralBleachingDatasetFromMetadata(Dataset):
    def __init__(self, metadata_df, root, transform=None):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata for this dataset.
            root (str): Root directory of the dataset.
            transform (callable, optional): Transform to apply to images.
        """
        self.metadata_df = metadata_df
        self.root = root
        self.transform = transform
        self.label_map = {"healthy": 0, "bleached": 1, "dead": 2}
        
        # Process metadata into a usable format
        self.data = []
        for _, row in metadata_df.iterrows():
            image_name = row["name"]
            class_label = row["label"]
            source = row["source"]
            metadata_row = {
                "location": row.get("location"),
                "date": row.get("date"),
                "sst": row.get("SST@90th_HS")
            }
            
            # Construct image path
            image_path = os.path.join(root, source, class_label, image_name)
            if os.path.exists(image_path):
                self.data.append({
                    "image_path": image_path,
                    "label": class_label,
                    "metadata": metadata_row
                })
            else:
                print(f"Warning: Image not found: {image_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image_path"]
        label = sample["label"]
        metadata = sample["metadata"]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Encode label as integer
        label_encoded = self.label_map[label]
        
        return image, label_encoded, metadata



class CoralBleachingDataset(Dataset):
    def __init__(self, root, coralnet_sources, transform=None):
        """
        Args:
            root (str): Root directory of the dataset.
            coralnet_sources (list): List of CoralNet source names to include.
            transform (callable, optional): Transform to apply to images.
        """
        self.root = root
        self.coralnet_sources = coralnet_sources
        self.transform = transform
        self.data = []
        self.label_map = {"healthy": 0, "bleached": 1, "dead": 2}  # Map labels to integers
        
        for source in coralnet_sources:
            source_path = os.path.join(root, source)
            metadata_path = os.path.join(source_path, "metadata.csv")
            
            # Load metadata
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata CSV not found in source: {source}")
            metadata = pd.read_csv(metadata_path)
            
            # Loop through possible label folders
            for label in self.label_map.keys():
                label_folder = os.path.join(source_path, label)
                if not os.path.exists(label_folder):
                    continue  # Skip if the label folder doesn't exist
                
                # Process images within the label folder
                for _, row in metadata.iterrows():
                    image_name = row["name"]
                    class_label = row["label"]
                    
                    # Ensure the label matches the folder name
                    if class_label != label:
                        continue
                    
                    metadata_row = {
                        "location": row.get("location"),
                        "date": row.get("date"),
                        "sst": row.get("SST@90th_HS")
                    }
                    
                    # Construct image path
                    image_path = os.path.join(label_folder, image_name)
                    if os.path.exists(image_path):
                        self.data.append({
                            "image_path": image_path,
                            "label": class_label,
                            "metadata": metadata_row
                        })
                    else:
                        print(f"Warning: Image not found: {image_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image_path"]
        label = sample["label"]
        metadata = sample["metadata"]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Encode label as integer
        label_encoded = self.label_map[label]
        
        return image, label_encoded, metadata


class CoralBleachingDatasetWithSST(Dataset):
    def __init__(self, metadata_df, root, transform=None):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata for this dataset.
            root (str): Root directory of the dataset.
            transform (callable, optional): Transform to apply to images.
        """
        self.metadata_df = metadata_df
        self.root = root
        self.transform = transform
        self.label_map = {"healthy": 0, "bleached": 1, "dead": 2}
        
        # Process metadata into a usable format
        self.data = []
        for _, row in metadata_df.iterrows():
            image_name = row["name"]
            class_label = row["label"]
            source = row["source"]
            metadata_row = {
                "location": row.get("location"),
                "date": row.get("date"),
                "sst": row.get("SST@90th_HS", 16.0)
            }
            
            # image path
            image_path = os.path.join(root, source, class_label, image_name)
            if os.path.exists(image_path):
                self.data.append({
                    "image_path": image_path,
                    "label": class_label,
                    "metadata": metadata_row
                })
            else:
                print(f"Warning: Image not found: {image_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image_path"]
        label = sample["label"]
        metadata = sample["metadata"]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Encode label as integer
        label_encoded = self.label_map[label]

        # Extract SST value as tnsor
        sst = metadata["sst"]
        sst_tensor = torch.tensor([sst], dtype=torch.float32)

        return image, sst_tensor, label_encoded


def split_dataset(root, coralnet_sources, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, transform=None):
    """
    Split the dataset into train, validation, and test sets with stratified sampling.

    Args:
        root (str): Root directory of the dataset.
        coralnet_sources (list): List of CoralNet source names to include.
        train_ratio (float): Proportion of the dataset to use for training.
        val_ratio (float): Proportion of the dataset to use for validation.
        test_ratio (float): Proportion of the dataset to use for testing.
        transform (callable, optional): Transform to apply to images.
    
    Returns:
        train_dataset, val_dataset, test_dataset: Dataset objects for training, validation, and testing.
    """
    # Combine metadata from all CoralNet sources
    all_metadata = []
    for source in coralnet_sources:
        metadata_path = os.path.join(root, source, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata CSV not found in source: {source}")
        
        metadata = pd.read_csv(metadata_path)
        metadata["source"] = source  # Add source column for reference
        all_metadata.append(metadata)
    
    # Concatenate all metadata into a single DataFrame
    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    
    # Stratified split into Train/Val/Test
    X = combined_metadata  # Data (everything including image names)
    y = combined_metadata["label"]  # Labels (class labels)
    
    # First split: Train and Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=42
    )
    
    # Second split: Val and Test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)  # Adjusted ratio for the second split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio_adjusted), stratify=y_temp, random_state=42
    )
    
    # Create Dataset objects
    train_dataset = CoralBleachingDatasetFromMetadata(X_train, root, transform)
    val_dataset = CoralBleachingDatasetFromMetadata(X_val, root, transform)
    test_dataset = CoralBleachingDatasetFromMetadata(X_test, root, transform)
    
    return train_dataset, val_dataset, test_dataset


def split_dataset_with_sst(root, coralnet_sources, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, transform=None):
    """
    Split the dataset into train, validation, and test sets with stratified sampling.

    Args:
        root (str): Root directory of the dataset.
        coralnet_sources (list): List of CoralNet source names to include.
        train_ratio (float): Proportion of the dataset to use for training.
        val_ratio (float): Proportion of the dataset to use for validation.
        test_ratio (float): Proportion of the dataset to use for testing.
        transform (callable, optional): Transform to apply to images.
    
    Returns:
        train_dataset, val_dataset, test_dataset: Dataset objects for training, validation, and testing.
    """
    # Combine metadata from all CoralNet sources
    all_metadata = []
    for source in coralnet_sources:
        metadata_path = os.path.join(root, source, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata CSV not found in source: {source}")
        
        metadata = pd.read_csv(metadata_path)
        metadata["source"] = source  # Add source column for reference
        all_metadata.append(metadata)
    
    # Concatenate all metadata into a single DataFrame
    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    
    # Stratified split into Train/Val/Test
    X = combined_metadata  # Data (everything including image names)
    y = combined_metadata["label"]  # Labels (class labels)
    
    # First split: Train and Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=42
    )
    
    # Second split: Val and Test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)  # Adjusted ratio for the second split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio_adjusted), stratify=y_temp, random_state=42
    )
    
    # Create Dataset objects
    train_dataset = CoralBleachingDatasetWithSST(X_train, root, transform)
    val_dataset = CoralBleachingDatasetWithSST(X_val, root, transform)
    test_dataset = CoralBleachingDatasetWithSST(X_test, root, transform)
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Example usage:
    # Define root directory and CoralNet sources
    root_dir = "../data/outputs/images"
    sources = ["WAPA Interns", "WAPA CoralNet Training", "WAPA Coral Inventory 2.0", 
               "CuraCao Coral Reef Assessment 2023 CUR", "Curacao Coral Reef Assessment 2023 ARU", "Altieri Biscayne Bay"]

    # Define transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset, val_dataset, test_dataset = split_dataset_with_sst(
        root=root_dir, coralnet_sources=sources, transform=data_transforms
    )

    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Validation Dataset: {len(val_dataset)} samples")
    print(f"Test Dataset: {len(test_dataset)} samples")

    # Example data sample
    image, sst, label = train_dataset[0]
    print("Image Shape:", image.shape)
    print("SST:", sst)
    print("Label:", label)