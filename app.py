from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model('eggplant.keras')

# Define class names
class_names = ['Healthy Leaf', 'Insect Pest Disease', 'Leaf Spot Disease', 
               'Mosaic Virus Disease', 'Small Leaf Disease', 
               'White Mold Disease', 'Wilt Disease']

app = FastAPI()

# Preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    # Preprocess the image and make a prediction
    img_array = preprocess_image(file_location)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    # Remove the temporary file
    os.remove(file_location)

    # Return the prediction as JSON
    return JSONResponse(content={"predicted_class": predicted_class, "confidence": float(confidence)})