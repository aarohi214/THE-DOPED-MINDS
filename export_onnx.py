import torch
import torch.onnx
from model import SimpleCNN
import os
import onnx
import onnxruntime as ort
import numpy as np

def export_model(model_path="best_model.pth", output_path="model.onnx"):
    print(f"Loading model from {model_path}...")
    
    # 1. Load Model
    # We need to know the number of classes. 
    # In train.py we saw classes: ['Center', 'Donut', 'Edge Local', 'Edge Ring', 'Local', 'Scratch', 'near full', 'none', 'random'] (9 classes)
    model = SimpleCNN(num_classes=9)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    else:
        print(f"Error: Model file {model_path} not found.")
        return

    # 2. Define Dummy Input
    # Shape: [Batch_Size, Channels, Height, Width]
    # We used 96x96 in training
    dummy_input = torch.randn(1, 3, 96, 96, requires_grad=True)

    # 3. Export to ONNX
    print(f"Exporting to {output_path}...")
    torch.onnx.export(model,               # model being run
                      dummy_input,         # model input (or a tuple for multiple inputs)
                      output_path,         # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    
    print("Export complete.")
    
    # 4. Verify ONNX Model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed.")
    
    # 5. Test Inference with ONNX Runtime
    print("Testing inference with ONNX Runtime...")
    ort_session = ort.InferenceSession(output_path)
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    torch_out = model(dummy_input)
    
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == "__main__":
    export_model()
