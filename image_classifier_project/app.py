from flask import Flask, render_template, request, send_from_directory
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('model/image_model.h5')
class_names = sorted(os.listdir('dataset'))

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return render_template('index.html', result="No image uploaded")

    # Save uploaded image to static/uploads
    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Read and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0]
    class_index = np.argmax(prediction)
    predicted_class = class_names[class_index]
    confidence = round(prediction[class_index] * 100, 2)

    return render_template(
        'index.html',
        result=predicted_class,
        confidence=confidence,
        image_url=f"/static/uploads/{filename}"
    )

# To serve uploaded images (Flask handles this automatically)
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
