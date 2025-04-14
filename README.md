# License_Plate_OCR
a flask-server for the graduation project , it aims to detect the Egyption license plates from a camera feed and it extracts the letters & numbers from it.
it takes an image as an input(post http-request) and responds with the result of the ocr
Used a YOLO model (which we trained) , OpenCV and PaddleOCR .

ToDos:
- enhance the performance of the whole process
- clean the code a little bit
- increase the ocr accuracy

How it works?
first the yolo model detects the license plate then we rotate the image if the license plate is not horizontal.
after rotating the image to make sure the license plate is horizontal we reapply yolo to get a more accurate bounding box/position of the licnese plate 
then we crop the image and feed the ocr model with the cropped image containing only the license plate 
