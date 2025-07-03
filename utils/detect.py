from ultralytics import YOLO
import os

# Load the model once globally to avoid repeated load
model = YOLO("yolov8n.pt")  # you can change to 'yolov8s.pt' for better accuracy

def detect_objects(image_path, save_dir="outputs"):
    """
    Runs YOLOv8 on the image and returns the detected object names.
    Also saves the annotated image to the outputs folder.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist.")

    results = model(image_path, save=True, project=save_dir, name="detect_results", exist_ok=True)

    # Extract object names
    labels = results[0].boxes.cls.tolist()
    names = results[0].names
    detected_objects = [names[int(cls_id)] for cls_id in labels]

    print(f"[INFO] Detected objects: {detected_objects}")
    return list(set(detected_objects))  # remove duplicates
