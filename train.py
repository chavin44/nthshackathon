from ultralytics import YOLO
import torch
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU at cuda:0: {gpu_name}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO("yolov8n.pt")

def train_model():
    model.train(
        data="e_waste_dataset/data.yaml", 
        epochs=50,  
        batch=8, 
        imgsz=640,
        workers=4,
        device=device,  
        pretrained=True,
        optimizer="auto",  
        save=True, 
        project="runs",  
        name="train_e_waste_fixed", 
        freeze=None,  
        amp=True,  
        val=True, 
    )
if __name__ == "__main__":
    train_model()
