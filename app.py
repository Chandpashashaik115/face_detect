# app.py

from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load pre-trained face detection model
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel'
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    color = request.form.get('color', 'r')  # Default to 'r' if not selected

    # Read the uploaded image
    image_data = file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process image based on selected color
    if color == 'r':
        img = cv2.merge([img[:, :, 0], np.zeros_like(img[:, :, 1]), np.zeros_like(img[:, :, 2])])
    elif color == 'g':
        img = cv2.merge([np.zeros_like(img[:, :, 0]), img[:, :, 1], np.zeros_like(img[:, :, 2])])
    elif color == 'b':
        img = cv2.merge([np.zeros_like(img[:, :, 0]), np.zeros_like(img[:, :, 1]), img[:, :, 2]])
    elif color == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color == 'face':
        detect_faces(img)

    # Convert the image with rectangles back to base64 for displaying
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return render_template('index.html', result_img=img_str)

def detect_faces(img):
    # Detect faces using the pre-trained face detection model
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    faces = net.forward()

    # Draw rectangles around detected faces
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:  # Adjust confidence threshold as needed
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

if __name__ == '__main__':
    app.run(debug=True)
