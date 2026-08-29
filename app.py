from pathlib import Path
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from cv2 import dnn_superres
from paddleocr import PaddleOCR

BASE_DIR = Path(__file__).parent

# Super-resolution (EDSR x4)
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(str(BASE_DIR / "EDSR_x4.pb"))
sr.setModel("edsr", 4)

# YOLO plate detector
model = YOLO(str(BASE_DIR / "best.pt"))

# PaddleOCR - load once globally (was per-request before, very slow)
ocr = PaddleOCR(use_angle_cls=True, lang='ar')


def preprocess_image(image):
    """
    Crops image to 10:3 aspect ratio (Egyptian plate approx) and grayscales.
    Keeps bottom strip where plate typically appears, then grayscales.
    """
    height, width = image.shape[:2]
    new_height = int(width * (3 / 10))
    y_start = max(0, height - new_height)
    cropped_image = image[y_start:, :]
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    return gray


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


def crop_plate_with_margin(image, box, margin_ratio=0.5):
    """Crops a detected license plate with a margin, ensuring it stays within image bounds."""
    x1, y1, x2, y2 = map(int, box.tolist())
    margin = int(margin_ratio * (y2 - y1))
    x1, y1, x2, y2 = max(0, x1 - margin), max(0, y1 - margin), min(image.shape[1], x2 + margin), min(image.shape[0], y2 + margin)
    return image[y1:y2, x1:x2]


def action(image):
    file_bytes = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
    results = model(image, verbose=False)[0]

    if len(results.boxes) == 0:
        return []

    # Process detected plates - use first detection
    for box in results.boxes.xyxy:
        plate = crop_plate_with_margin(image, box)
        if plate.size == 0:
            continue
        gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 200)

        # Detect skew via Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for line in lines for x1, y1, x2, y2 in [line[0]]]
            median_angle = float(np.median(angles)) if angles else 0
        else:
            median_angle = 0

        # Rotate whole image and re-detect for tighter box
        rotated_image = rotate_image(image, median_angle)
        rotated_results = model(rotated_image, verbose=False)[0]

        if len(rotated_results.boxes) == 0:
            # fallback: use deskewed plate directly without second crop
            neoPlate = preprocess_image(rotate_image(plate, median_angle))
        else:
            neoPlate = crop_plate_with_margin(rotated_image, rotated_results.boxes.xyxy[0], -0.05)
            neoPlate = preprocess_image(neoPlate)

        if neoPlate.size == 0:
            return []

        cv2.imwrite(str(BASE_DIR / "rec_neoplate_before.jpg"), neoPlate)
        neoPlate = cv2.cvtColor(neoPlate, cv2.COLOR_GRAY2BGR)

        # Super-resolve once (x4). Double 16x was too slow and produced huge images.
        result = sr.upsample(neoPlate)

        cv2.imwrite(str(BASE_DIR / "rec_neoplate_after.jpg"), result)
        ocred = ocr.ocr(result, cls=True)

        if not ocred or not ocred[0]:
            return []

        texts = [line[1][0] for line in ocred[0]]
        return texts

    return []


app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    print(f"Received image: {image.filename}")
    result_texts = action(image)
    return jsonify({'message': result_texts})


if __name__ == '__main__':
    app.run(debug=False)
