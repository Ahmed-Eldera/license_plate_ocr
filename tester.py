from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from cv2 import dnn_superres
from paddleocr import PaddleOCR
import os
import time
import contextlib
import io
import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)  # or logging.CRITICAL to suppress more


def silent_yolo_inference(model, image):
    # Suppress stdout temporarily
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        start_time = time.time()
        results = model(image)[0]  # Run inference
        end_time = time.time()
    inference_time = end_time - start_time
    return results, inference_time

sr = dnn_superres.DnnSuperResImpl_create()

# # Set the model type and the scale
# sr.setModel("lapsrn", 8)
path = "EDSR_x4.pb"
sr.readModel(path)
 
# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 4)
model = YOLO(r"./best.pt")
def preprocess_image(image, target_size=(1024, 768)):
    """
    Crops the top portion of the image to achieve a 10:3 aspect ratio,
    then resizes it to the target size (1024x768).
    Applies preprocessing steps to enhance OCR accuracy:
    - Converts to grayscale
    - Crops the image to maintain aspect ratio (10:3)
    - Resizes while maintaining aspect ratio
    - Adds padding if needed to match the target size

    :param image: Input image (numpy array)
    :param target_size: Tuple (width, height), default (1024, 768)
    :return: Preprocessed image ready for OCR
    """
    target_width, target_height = target_size
    height, width = image.shape[:2]

    # Step 1: Compute new height for 10:3 ratio
    new_height = int(width * (3 / 10))  # h = w * (3/10)

    # Step 2: Crop the image from the top (y: 0 → new_height)
    height = height-new_height
    cropped_image = image[height:, :]

    # Step 3: Resize while maintaining aspect ratio
    # resized_image = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Step 4: Convert to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    return gray

def sharpen_image(image):
    """
    Sharpens an image using a kernel.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The sharpened image.
    """
    # Define sharpening kernel
    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    # Apply the sharpening filter
    sharpened = cv2.filter2D(image, -1, sharpening_kernel)

    return sharpened
def ocr_preprocessing(image):
    """Preprocesses an image for OCR by converting it to grayscale and applying adaptive thresholding."""
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    processed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 4)
    return processed
def rotate_image(image, angle):
    """Rotates an image without cropping it."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
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
    x1, y1, x2, y2 = max(0, x1 - margin), max(0, y1 - margin), min(image.shape[1], x2 + margin), min(image.shape[0], y2 + margin)
    return image[y1:y2, x1:x2]
def draw_ocr_results(image, ocr_results):
    """Draws OCR bounding boxes and text on an image."""
    output_image = image.copy()
    for bbox, text, confidence in ocr_results:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
        x1, y1, x2, y2, x3, y3, x4, y4 = map(int, [x1, y1, x2, y2, x3, y3, x4, y4])
        cv2.polylines(output_image, [np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])], True, (0, 255, 0), 2)
        cv2.putText(output_image, f"{text} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return output_image

def action(image):
    ocr = PaddleOCR(use_angle_cls=True, lang='ar') 

    file_bytes = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start_time = time.time()
    # Run YOLO inference
    results, inf_time = silent_yolo_inference(model, image)

    # Log inference time
    with open("yolo_times.txt", "a") as f:
        f.write(f"Inference time: {inf_time:.4f} seconds\n")

    # image_with_boxes = draw_bounding_boxes(image, results.boxes.xyxy)

    # Process detected plates
    for box in results.boxes.xyxy:
        plate = crop_plate_with_margin(image, box)
        gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 200)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for line in lines for x1, y1, x2, y2 in [line[0]]]
            median_angle = np.median(angles) if angles else 0
            # print(f"Detected Angle: {median_angle}")

            # Rotate the plate and re-run YOLO
            rotated_plate = rotate_image(plate, median_angle)
            rotated_image = rotate_image(image, median_angle)
            rotated_results,x = silent_yolo_inference(model, rotated_image)
            with open("yolo_times.txt", "a") as f:
                f.write(f"Inference time: {x:.4f} seconds\n")
            # rotated_yolo_image = draw_bounding_boxes(rotated_image, rotated_results.boxes.xyxy)
            neoPlate = crop_plate_with_margin(rotated_image, rotated_results.boxes.xyxy[0], -0.05)
            neoPlate = preprocess_image(neoPlate)
            # cv2.imwrite("./rec_neoplate_before.jpg",neoPlate)
            neoPlate = cv2.cvtColor(neoPlate, cv2.COLOR_GRAY2BGR)
            # result = sr.upsample(neoPlate)
            # result = sr.upsample(result)
            # result = sharpen_image(result)
            # cv2.imwrite("./rec_neoplate_after.jpg",neoPlate)
            ocr_start= time.time()
            ocred= ocr.ocr(neoPlate, cls=True)
            ocr_end = time.time()
            ocr_time = ocr_start - ocr_end
            open("ocr_time.txt", "a").write(f"{ocr_time:.4f} \n")
            texts = [line[1][0] for line in ocred[0]]
            end_time = time.time()
            whole_time = end_time - start_time
            open("whole_time.txt", "a").write(f"{whole_time:.4f} \n")

            return texts


def process_images_in_directory(directory_path, output_file_path):
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    
    with open(output_file_path, "a", encoding="utf-8") as outfile:
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(supported_extensions):
                image_path = os.path.join(directory_path, filename)
                try:
                    with open(image_path, "rb") as img_file:
                        print(f"Processing {filename}...")
                        texts = action(img_file)
                        text_output = " | ".join(texts) if texts else "No text found"
                        outfile.write(f"{filename}: {text_output}\n")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    outfile.write(f"{filename}: Error - {e}\n")

# Example usage
if __name__ == "__main__":
    input_dir = "images"  # Replace with your directory path
    output_txt = "paddle_no_super_res.txt"
    process_images_in_directory(input_dir, output_txt)
