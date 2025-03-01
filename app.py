from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Load YOLO model (Make sure the model path is correct)
model = YOLO(r"C:\Users\hotoe\Desktop\new models\best.pt")

# Load image
image_path = r"C:\Users\hotoe\Desktop\civic.jpg"  # Replace with your image path
image = Image.open(image_path)

# Run inference
results = model(image_path)[0]  # Get the first detection result

# Initialize drawing
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()  # Default font, can be replaced with a TTF font

# Loop through detections
for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
    x1, y1, x2, y2 = box.tolist()
    confidence = conf.item()

    # Get class label (if available)
    class_id = int(cls.item())  # Convert to integer
    class_name = model.names[class_id] if model.names else f"ID {class_id}"

    # Draw bounding box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Display class name and confidence
    label = f"{class_name}: {confidence:.2f}"
    text_size = draw.textbbox((x1, y1), label, font=font)  # Get text bounding box
    draw.rectangle([text_size[0], text_size[1], text_size[2], text_size[3]], fill="red")  # Background for text
    draw.text((x1, y1), label, fill="white", font=font)  # Write text

# Show the image with detections
image.show()
