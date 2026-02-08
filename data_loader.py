import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import glob

class WM811KDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            # Support common image extensions
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.samples.extend(glob.glob(os.path.join(class_dir, ext)))
                # Also check uppercase extensions just in case
                self.samples.extend(glob.glob(os.path.join(class_dir, ext.upper())))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        # Label is directory name
        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[class_name]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error (or handle differently)
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, num_workers=0):
    # Define transforms
    # Resize to 96x96 for speed (<10ms target), can increase if valid
    # Start with 224x224 as commonly used, but might need downscaling for speed
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = WM811KDataset(data_dir, transform=train_transform) # Apply train transform initially
    
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Overwrite transform for validation set (hacky but works for random_split)
    # Ideally should use subset with different transform, but for simplicity:
    val_dataset.dataset.transform = val_transform 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, dataset.classes

if __name__ == "__main__":
    # Test the loader
    import kagglehub
    
    # Hardcoded path for testing, should match download_dataset.py output location
    # Note: user path might vary, but we can try to find it or pass it.
    # For now, let's assume the path we found earlier.
    path = r"C:\Users\alokk\.cache\kagglehub\datasets\muhammedjunayed\wm811k-silicon-wafer-map-dataset-image\versions\1\WM811k_Dataset"
    
    if os.path.exists(path):
        train_loader, val_loader, classes = get_dataloaders(path, batch_size=4)
        print(f"Classes: {classes}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
    else:
        print(f"Path not found: {path}")
