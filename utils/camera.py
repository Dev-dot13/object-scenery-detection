import cv2
import os
from datetime import datetime

def capture_image(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Camera not accessible.")

    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to capture image from camera.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"capture_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"[INFO] Image saved to {filename}")

    cap.release()
    cv2.destroyAllWindows()
    return filename
