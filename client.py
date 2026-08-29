import requests

import sys
from pathlib import Path

url = 'http://127.0.0.1:5000/upload'
image_path = sys.argv[1] if len(sys.argv) > 1 else './A_hamdy.png'

if not Path(image_path).exists():
    print(f"Image not found: {image_path}")
    sys.exit(1)

with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)
    response.raise_for_status()
    print('Response from server:', response.json())
