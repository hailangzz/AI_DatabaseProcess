import kagglehub

# Download latest version
path = kagglehub.dataset_download("zanellar/electric-wires-image-segmentation")

print("Path to dataset files:", path)