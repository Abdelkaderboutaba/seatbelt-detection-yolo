from ultralytics import YOLO
import shutil
import os

DATA_YAML = "data.yaml"

def save_best_model():
    best = "runs/seatbelt_detection/weights/best.pt"
    if os.path.exists(best):
        shutil.copy(best, "best.pt")
        print("Model saved locally.")

def train():
    model = YOLO("yolov8s.pt")

    model.train(
        data=DATA_YAML,
        epochs=40,
        imgsz=1280,
        batch=8,
        name="seatbelt_detection",
        project="runs",
        device=0
    )

    save_best_model()

if __name__ == "__main__":
    train()