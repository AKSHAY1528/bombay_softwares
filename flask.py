from flask import Flask, request, render_template, jsonify
import cv2
import os
import numpy as np
from joblib import load  # Import joblib for scikit-learn models

app = Flask(__name__)

# Load your pre-trained model (replace with your actual filename)
model = load('my_model.pkl')  # Replace with your model filename

# Function to preprocess an image and extract features
def preprocess_image(image_path, target_size=(100, 100)):
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        print(f"Error loading image: {image_path}")
        return None

    # Resize the image
    image = cv2.resize(image, target_size)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)

    return equalized

@app.route('/')
def upload_form():
    return render_template('upload_image.html')

@app.route('/', methods=['POST'])
def handle_upload():
    if 'photo' not in request.files:
        return render_template('upload_image.html', message='No photo selected')

    photo = request.files['photo']
    photo.save('photo.jpg')  # Save the uploaded image

    # Read the image and preprocess it
    img = cv2.imread('photo.jpg')
    img = preprocess_image('photo.jpg')  # Preprocess the image

    if img is None:
        return jsonify({'error': 'Failed to preprocess the image'})

    # Add a batch dimension for prediction (if your model expects it)
    if len(img.shape) == 2:  # Check if image is grayscale
        img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction using the model
    prediction = model.predict(img)  # Assuming model predicts class labels

    # Get the predicted class (assuming model returns class labels)
    predicted_class = prediction[0]  # Access the predicted class label

    # You can customize the response based on your needs
    response = {
        'predicted_class': str(predicted_class),
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
