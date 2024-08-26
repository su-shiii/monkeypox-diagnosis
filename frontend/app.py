from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import time 

app = Flask(__name__)

# Load the model
model = load_model('model/model.keras')

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    # Read the image file as a byte stream
    img = image.load_img(io.BytesIO(file.read()), target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = 'Monkeypox' if prediction[0] > 0.5 else 'Not Monkeypox'
    
    # Pause for 1 second before showing result- this is for my own peace of mind tbh
    time.sleep(1)
    
    return render_template('result.html', prediction=predicted_class)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
