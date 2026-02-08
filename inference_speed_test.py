import torch
import time
from model import SimpleCNN
from PIL import Image
from torchvision import transforms
import os

# Define inference function
def run_inference_test(model_path, image_path=None, num_runs=100):
    # Load Model
    device = torch.device("cpu") # CPU is fast enough < 10ms
    model = SimpleCNN(num_classes=8) # 8 classes for SEM dataset
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((96, 96)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dummy input or load real image
    if image_path and os.path.exists(image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading image: {e}")
            return
    else:
        print("Using dummy input (random noise).")
        input_tensor = torch.randn(1, 3, 96, 96).to(device)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)

    # Benchmark
    print(f"Running benchmark for {num_runs} iterations...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end_time = time.perf_counter()

    avg_latency = (end_time - start_time) / num_runs * 1000 # ms
    throughput = num_runs / (end_time - start_time) # img/s

    print(f"\nResults:")
    print(f"Device: {device}")
    print(f"Average Latency: {avg_latency:.4f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")
    
    if avg_latency < 10:
        print("\n[SUCCESS] Latency is under 10 ms!")
    else:
        print("\n[WARNING] Latency is over 10 ms.")

if __name__ == "__main__":
    model_path = "best_sem_model.pth"
    # Optional: Test with a real image from the dataset if available
    # Check if we can find a sample image
    sample_image = None
    dataset_root = r"C:\Users\alokk\Downloads\DATASET main\DATASET"
    if os.path.exists(dataset_root):
        # Find first jpg/png
        for root, dirs, files in os.walk(dataset_root):
             for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_image = os.path.join(root, file)
                    break
             if sample_image: break
    
    if sample_image:
        print(f"Testing with image: {sample_image}")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
    else:
        run_inference_test(model_path, sample_image)
