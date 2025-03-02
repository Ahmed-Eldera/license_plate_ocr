import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def detect_hough_lines(image):
    """Detects and overlays Hough lines on the given image."""
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for visualization
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines
    return output


def rotate_image(image, angle):
    """Rotates an image without cropping it."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the new bounding dimensions after rotation
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the matrix for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated


# Load YOLO model
model = YOLO(r"C:\Users\hotoe\Desktop\new models\best.pt")

# Load image
image_path = r"C:\Users\hotoe\Desktop\de.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

# Run YOLO inference
results = model(image)[0]

# Draw bounding boxes on the original image for visualization
image_with_boxes = image.copy()
for box in results.boxes.xyxy:
    x1, y1, x2, y2 = map(int, box.tolist())

    # Draw bounding box on original image
    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box

    # Calculate margin (25% of bounding box height)
    margin = int(0.5 * (y2 - y1))

    # Apply margin while ensuring it doesn't go out of image bounds
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.shape[1], x2 + margin)
    y2 = min(image.shape[0], y2 + margin)

    # Crop detected license plate with margin
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

            # Rotate both the cropped plate and the entire image
            rotated_plate = rotate_image(plate, median_angle)
            rotated_image = rotate_image(image, median_angle)

            # Run YOLO again on the rotated plate
            rotated_results = model(rotated_image)[0]

            for rotated_box in rotated_results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, rotated_box.tolist())
                cv2.rectangle(rotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box

            # Enhance the image
            gray_rotated = cv2.cvtColor(rotated_plate, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray_rotated, (9, 9), 1.5)  # Blur the image
            sharpened = cv2.addWeighted(gray_rotated, 3, blurred, -2, 0)  # Add contrast

            gray_eq = cv2.equalizeHist(sharpened)
            lined_img = detect_hough_lines(gray_eq)
            _, binary = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Display images using Matplotlib
            fig, axes = plt.subplots(1, 6, figsize=(20, 5))

            axes[0].imshow(image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(image_with_boxes)
            axes[1].set_title("YOLO Detections (Before Cropping)")
            axes[1].axis("off")

            axes[2].imshow(plate)
            axes[2].set_title("Cropped Plate (With Margin)")
            axes[2].axis("off")

            axes[3].imshow(rotated_image)
            axes[3].set_title("Rotated Plate (YOLO Again)")
            axes[3].axis("off")

            axes[4].imshow(lined_img, cmap="gray")
            axes[4].set_title("Hough Transform")
            axes[4].axis("off")

            axes[5].imshow(binary, cmap="gray")
            axes[5].set_title("Binarized Plate")
            axes[5].axis("off")

            plt.show()