import torch
import time
import numpy as np
from PIL import Image
from model import SimpleCNN
import torchvision.transforms as transforms

class DefectDetector:
    def __init__(self, use_cuda=False, model_path="best_sem_model.pth"):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Initializing DefectDetector on {self.device}...")
        
        # Classes for SEM dataset
        self.classes = ['Bridge', 'CMP scratch', 'Clean', 'LER', 'crack', 'manforsed via', 'open', 'short']
        num_classes = len(self.classes)
        
        self.model = SimpleCNN(num_classes=num_classes)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using random weights.")

        self.model.to(self.device)
        self.model.eval()
        
        # Transform must match training (resize to 96x96)
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Dummy warmup
        dummy_input = torch.randn(1, 3, 96, 96).to(self.device)
        self.model(dummy_input)
        print("Model initialized and warmed up.")

    def preprocess(self, images):
        """
        Convert list of numpy images (or PIL images) to tensor batch.
        """
        # Images might be numpy arrays (H, W, C) from main.py generator
        # We need to convert them to PIL for the transform, or adapt transform
        
        batch_tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            # Helper to convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            batch_tensors.append(self.transform(img))
            
        return torch.stack(batch_tensors).to(self.device)

    def detect_batch(self, images):
        """
        Run detection on a batch of images.
        """
        if not images:
            return []

        start_time = time.perf_counter()
        
        # Preprocess
        input_tensor = self.preprocess(images)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
        inference_time = (time.perf_counter() - start_time)
        per_image_time = inference_time / len(images)
        
        results = []
        cpu_preds = predictions.cpu().numpy()
        cpu_probs = probs.cpu().numpy()
        
        for i, pred_idx in enumerate(cpu_preds):
            pred_class = self.classes[pred_idx]
            score = float(cpu_probs[i][pred_idx])
            
            # Assume 'none' means no defect
            has_defect = (pred_class != 'none')
            
            results.append({
                "image_idx": i,
                "has_defect": has_defect,
                "confidence": score,
                "latency_sec": per_image_time,
                "defect_type": pred_class
            })
            
        return results
