import requests

url = 'http://127.0.0.1:5000/upload'
image_path = './A_hamdy.png'  # Replace with your image path

with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)

print('Response from server:', response.json())
