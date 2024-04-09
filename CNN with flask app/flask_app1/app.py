from flask import Flask, render_template, request, session, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = 'some_secret_key'  # for using session

# Load the trained model
model = load_model('mnist_cnn_model.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html', uploaded_image=session.get('uploaded_image'), result=session.get('result'))

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file:
        image_path = "static/uploaded_image.png"
        file.save(image_path)
        session['uploaded_image'] = image_path
        return redirect(url_for('index'))
    return "Failed to upload image!"

@app.route('/predict', methods=['POST'])
def predict_image():
    expected_class = request.form.get('expected_class')
    image_path = session.get('uploaded_image')
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))
    image_arr = np.asarray(image) / 255.0  # Normalize
    image_arr = image_arr.reshape(1, 28, 28, 1)  # Reshape for the model
    predictions = model.predict(image_arr)
    predicted_class = np.argmax(predictions)
    
    session['result'] = {
        'expected': expected_class,
        'predicted': str(predicted_class)
    }
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
