import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()
        # Input: 3 x 96 x 96
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 48x48
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # pool -> 24x24
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # pool -> 12x12
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # pool -> 6x6
        
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Test model shape and inference speed estimate
    import time
    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 96, 96)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
        
    start = time.perf_counter()
    for _ in range(100):
        _ = model(dummy_input)
    end = time.perf_counter()
    
    print(f"Avg Inference Time (CPU): {(end - start) / 100 * 1000:.4f} ms")
    
    # Check parameter count
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {pytorch_total_params}")
