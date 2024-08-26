import joblib
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

# Load your model (example for a trained SVM model)
model = joblib.load('model.pkl')

def classify_image(image_path):
    image = Image.open(image_path)
    image = image.resize((128, 128))  # Resize if necessary
    image_array = np.array(image).flatten().reshape(1, -1)
    
    # If using a scaler, apply it to the image array
    # scaler = StandardScaler().fit(image_array)
    # image_array = scaler.transform(image_array)
    
    prediction = model.predict(image_array)
    return str(prediction[0])
