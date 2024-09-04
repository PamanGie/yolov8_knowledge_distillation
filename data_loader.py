import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

class YOLODataset(Dataset):
    def __init__(self, images_folder, labels_folder, transform=None):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.image_filenames = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpg')])
        self.transform = transform

        # Print jumlah gambar dan file yang ditemukan untuk debugging
        print(f"Found {len(self.image_filenames)} images in {images_folder}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_folder, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")

        # Load corresponding label file (optional)
        label_path = os.path.join(self.labels_folder, self.image_filenames[idx].replace('.jpg', '.txt'))

        # Debug: Print paths to ensure files exist
        print(f"Loading image: {img_path}")
        print(f"Loading label: {label_path}")

        # Apply transformations (resizing, etc.)
        if self.transform:
            image = self.transform(image)

        # Return image and label path
        return image, label_path

# Dataset and DataLoader setup
def load_dataset(data_path, batch_size, transform):
    """
    Load dataset with custom YOLO dataset loader.
    """
    train_images_path = Path(data_path).parent / 'images/train'
    train_labels_path = Path(data_path).parent / 'labels/train'
    val_images_path = Path(data_path).parent / 'images/val'
    val_labels_path = Path(data_path).parent / 'labels/val'

    # Dataset untuk training dan validation
    train_dataset = YOLODataset(images_folder=train_images_path, labels_folder=train_labels_path, transform=transform)
    val_dataset = YOLODataset(images_folder=val_images_path, labels_folder=val_labels_path, transform=transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
