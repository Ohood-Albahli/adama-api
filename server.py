from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import numpy as np
import tensorflow as tf
from Test_model import AdamaTester
from Adama_model import csv_path, data_dir

# Load the model once
tester = AdamaTester(csv_path, data_dir)

# Class names must match what your model was trained on
CLASS_NAMES = ["Psoriasis", "Eczema"]
THRESHOLD = 0.5

# Prepare Flask app
app = Flask(__name__)

def preprocess_image(base64_str):
    image_data = base64.b64decode(base64_str.split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((224, 224))  # match model input size
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 224, 224, 3)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_base64 = data.get('image')

    if not image_base64:
        return jsonify({'error': 'No image provided'}), 400

    try:
        preprocessed = preprocess_image(image_base64)
        prediction = tester.model.predict(preprocessed)
        top_idx = np.argmax(prediction[0])
        confidence = prediction[0][top_idx]
        label = CLASS_NAMES[top_idx] if confidence >= THRESHOLD else "Unknown"

        return jsonify({
            'diagnosis': label,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
