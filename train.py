from ultralytics import YOLO
import torch
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU at cuda:0: {gpu_name}")

# Check if CUDA is available and select the appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")  # Prints which device is being used (CPU or GPU)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
def train_model():
    model.train(
        data="e_waste_dataset/data.yaml",  # Path to dataset configuration
        epochs=50,  # Number of training epochs
        batch=8,  # Adjust based on GPU memory
        imgsz=640,  # Image size
        workers=4,
        device=device,  # Use GPU (CUDA) if available
        pretrained=True,  # Use pretrained weights
        optimizer="auto",  # Default optimizer
        save=True,  # Save trained model
        project="runs",  # Directory to save runs
        name="train_e_waste_fixed",  # Experiment name
        freeze=None,  # Do not freeze layers
        amp=True,  # Automatic Mixed Precision for faster training
        val=True,  # Run validation after training
    )
if __name__ == "__main__":
    train_model()