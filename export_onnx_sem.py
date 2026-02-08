import torch
import torch.onnx
from model import SimpleCNN
import os
import sys

# Set default encoding to utf-8 for stdout/stderr to avoid charmap errors
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def export_model(model_path, onnx_path):
    # Initialize model with 8 classes for SEM dataset
    model = SimpleCNN(num_classes=8)
    
    # Load trained weights
    device = torch.device("cpu")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model not found at {model_path}")
        return

    # Create dummy input (batch_size=1, channels=3, height=96, width=96)
    dummy_input = torch.randn(1, 3, 96, 96)

    # Export
    print(f"Exporting to {onnx_path}...")
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("Export complete!")
    except Exception as e:
        print(f"Error during export: {e}")
        return

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully.")
    except ImportError:
        print("ONNX library not installed, skipping verification.")
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    model_path = "best_sem_model.pth"
    onnx_path = "model_sem.onnx"
    export_model(model_path, onnx_path)
