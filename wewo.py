from paddleocr import PaddleOCR

# Initialize the PaddleOCR instance (this will use the default model which supports multiple languages, including Arabic and English)
ocr = PaddleOCR(use_angle_cls=True, lang='ar')  # You can change 'en' to other language codes like 'ar' for Arabic

# List of image paths
image_paths = [
    # r"021.jpg",  # Path to your image
    r"01.jpg",
]

# Loop through each image and perform OCR
for img_path in image_paths:
    print(f"Processing image: {img_path}")

    # Perform OCR on the image
    result = ocr.ocr(img_path, cls=True)

    # Print the OCR output
    for line in result[0]:
        text = line[1][0]  # Extract the detected text
        print(f"Detected text: {text}")

    print("\n" + "-" * 50 + "\n")
