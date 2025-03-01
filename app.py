import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLO model
model = YOLO(r"C:\Users\hotoe\Desktop\new models\best.pt")

# Load image
image_path = r"C:\Users\hotoe\Desktop\der.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

# Run YOLO inference
results = model(image)[0]

for box in results.boxes.xyxy:
    x1, y1, x2, y2 = map(int, box.tolist())

    # Crop detected license plate
    plate = image[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 200)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)

    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))  # Calculate angle
            angles.append(angle)

        if angles:
            median_angle = np.median(angles)  # Get the most common angle
            print(f"Detected Angle: {median_angle}")

            # Get rotation matrix
            h, w = plate.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)

            # Rotate the plate
            rotated_plate = cv2.warpAffine(plate, M, (w, h))

            # Convert to RGB for Matplotlib
            rotated_plate_rgb = cv2.cvtColor(rotated_plate, cv2.COLOR_BGR2RGB)

            # Display images using Matplotlib
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(plate)
            axes[1].set_title("Cropped Plate")
            axes[1].axis("off")

            axes[2].imshow(rotated_plate_rgb)
            axes[2].set_title("Corrected Plate")
            axes[2].axis("off")

            plt.show()
