import kagglehub

# Download latest version
print("Downloading dataset...")
path = kagglehub.dataset_download("muhammedjunayed/wm811k-silicon-wafer-map-dataset-image")

print("Path to dataset files:", path)
