import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from data_loader import get_dataloaders
from model import SimpleCNN

def train_model(data_dir, num_epochs=5, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Dataset path not found at {data_dir}")
        return

    # Initialize data loaders
    # Using the existing get_dataloaders which handles the folder structure
    try:
        train_loader, val_loader, classes = get_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
        print(f"Data loaded successfully.")
        print(f"Classes found ({len(classes)}): {classes}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialize model
    model = SimpleCNN(num_classes=len(classes)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_sem_model.pth")
            print("Saved best model.")

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    # Path to the new SEM dataset
    dataset_path = r"C:\Users\alokk\Downloads\DATASET main\DATASET"
    
    train_model(dataset_path, num_epochs=args.epochs, batch_size=args.batch_size)
