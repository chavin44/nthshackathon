from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
