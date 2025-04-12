import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def rotate_image(image, angle):
    """Rotates an image without cropping it."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, M, (new_w, new_h))


def draw_bounding_boxes(image, boxes):
    """Draws bounding boxes on an image."""
    output_image = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
    return output_image


def crop_plate_with_margin(image, box, margin_ratio=0.5):
    """Crops a detected license plate with a margin, ensuring it stays within image bounds."""
    x1, y1, x2, y2 = map(int, box.tolist())

    margin = int(margin_ratio * (y2 - y1))
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.shape[1], x2 + margin)
    y2 = min(image.shape[0], y2 + margin)

    return image[y1:y2, x1:x2]  # Return cropped plate


# Load YOLO model
model = YOLO(r"C:\Users\hotoe\Desktop\new models\best.pt")

# Load image
image_path = r"C:\Users\hotoe\Desktop\mazen2.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run YOLO inference
results = model(image)[0]

# Draw bounding boxes
image_with_boxes = draw_bounding_boxes(image, results.boxes.xyxy)

# Process detected plates
for box in results.boxes.xyxy:
    plate = crop_plate_with_margin(image, box)

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
            median_angle = np.median(angles)
            print(f"Detected Angle: {median_angle}")

            # Rotate the plate
            rotated_plate = rotate_image(plate, median_angle)
            rotated_image = rotate_image(image,median_angle)
            # Run YOLO again on the rotated plate
            rotated_results = model(rotated_image)[0]
            rotated_yolo_image = draw_bounding_boxes(rotated_image, rotated_results.boxes.xyxy)
            neoPlate = crop_plate_with_margin(rotated_yolo_image,  rotated_results.boxes.xyxy[0],0)
            # Display images
            fig, axes = plt.subplots(1, 5, figsize=(20, 5))

            axes[0].imshow(neoPlate)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(image_with_boxes)
            axes[1].set_title("YOLO Detections (Before Cropping)")
            axes[1].axis("off")

            axes[2].imshow(plate)
            axes[2].set_title("Cropped Plate (With Margin)")
            axes[2].axis("off")

            axes[3].imshow(rotated_plate)
            axes[3].set_title("Rotated Plate")
            axes[3].axis("off")

            axes[4].imshow(rotated_yolo_image)
            axes[4].set_title("YOLO Detections (After Rotation)")
            axes[4].axis("off")

            plt.show()
