"""
train_model.py - YOLOv8 Pothole Detection Model Trainer

This script trains the YOLOv8 model on the Kaggle pothole dataset
and saves the best model as 'best.pt' in the project root.

HOW TO RUN:
    python train_model.py

After training, the model will be saved automatically.
The Flask app (app.py) will then load 'best.pt' for detection.
"""

from ultralytics import YOLO
import os
import shutil

# ---- Configuration ----
# How many training iterations (epochs) to run.
# Start with 20 for a fast result. Increase to 50+ for better accuracy.
EPOCHS = 20

# Batch size — how many images to process at once.
# Reduce to 8 if you get out-of-memory errors.
BATCH_SIZE = 16

# Image size for training (dataset is high-res, resize to 640 for speed)
IMAGE_SIZE = 640

# Training run name (results saved here inside 'runs/detect/')
RUN_NAME = "pothole_detector"

# Path to the dataset YAML file
DATA_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", "data.yaml")


def train():
    """
    Main training function.
    Loads a pretrained YOLOv8n (nano) model and fine-tunes it
    on the pothole dataset. YOLOv8n is fast and beginner-friendly.
    """

    print("=" * 60)
    print("  Smart Road Monitoring — YOLOv8 Training")
    print("=" * 60)
    print(f"\n  Dataset : {DATA_YAML}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Batch   : {BATCH_SIZE}")
    print(f"  ImgSize : {IMAGE_SIZE}")
    print("\n  Starting training... This may take a few minutes.\n")

    # Load YOLOv8 Nano — fastest model, ideal for beginners and limited hardware
    # 'yolov8n.pt' is a pretrained model on COCO. We fine-tune it on our pothole data.
    model = YOLO("yolov8n.pt")

    # ---- Start Training ----
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        name=RUN_NAME,
        exist_ok=True,      # Overwrite if same run name exists
        patience=10,        # Stop early if no improvement for 10 epochs
        verbose=True
    )

    print("\n✅ Training complete!")

    # ---- Copy best.pt to project root ----
    # After training, YOLO saves the best model inside 'runs/detect/<name>/weights/best.pt'
    best_pt_path = os.path.join("runs", "detect", RUN_NAME, "weights", "best.pt")

    if os.path.exists(best_pt_path):
        destination = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
        shutil.copy2(best_pt_path, destination)
        print(f"\n🎯 Model saved to: {destination}")
        print("✅ You can now run: python app.py")
    else:
        print(f"\n⚠️  Could not find best.pt at: {best_pt_path}")
        print("   Please check the 'runs/detect/' folder manually.")


if __name__ == "__main__":
    train()
