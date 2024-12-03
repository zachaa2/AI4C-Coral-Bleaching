import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

# Define label mapping
LABEL_MAP = {'healthy': 0, 'dead': 1, 'bleached': 2}
LABEL_NAMES = list(LABEL_MAP.keys())

class CoralBleachingDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None,
                 image_col='name', label_col='label', location_col='location',
                 date_col='date', watch_loca='CoralReefWatch location', tempe='SST@90th_HS'):
        self.root_dir = root_dir
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.location_col = location_col
        self.watch_loca = watch_loca
        self.tempe = tempe
        self.date_col = date_col

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Reading data
        img_name = self.metadata.iloc[idx][self.image_col]
        label_str = self.metadata.iloc[idx][self.label_col]
        location = self.metadata.iloc[idx][self.location_col]
        date = self.metadata.iloc[idx][self.date_col]
        tempe = self.metadata.iloc[idx][self.tempe]

        # Map label string to integer
        label = LABEL_MAP[label_str]

        img_path = os.path.join(self.root_dir, label_str, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Returns a dictionary containing images, labels, locations, and dates
        return {
            'image': image,
            'label': label,
            'location': location,
            'date': date,
            'SST@90th_HS': tempe
        }

def split_dataset(metadata_file, test_size=0.2, random_state=42):
    metadata = pd.read_csv(metadata_file)
    train_data, val_data = train_test_split(metadata, test_size=test_size, random_state=random_state)
    train_data.to_csv('train_metadata.csv', index=False)
    val_data.to_csv('val_metadata.csv', index=False)
    return 'train_metadata.csv', 'val_metadata.csv'

def collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'labels': torch.tensor([item['label'] for item in batch]),
        'locations': [item['location'] for item in batch],
        'dates': [item['date'] for item in batch],
        'SST@90th_HS': [item['SST@90th_HS'] for item in batch]
    }