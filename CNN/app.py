from flask import Flask, render_template, request
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = keras.models.load_model("dog_cat_cnn_model.h5")

# -----------------------------
# Home Route
# -----------------------------
@app.route('/')
def home():
    return render_template("index.html")

# -----------------------------
# Prediction Route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = model.predict(img_array)[0][0]

        if pred > 0.5:
            result = f"Dog ({pred*100:.2f}%)"
        else:
            result = f"Cat ({(1-pred)*100:.2f}%)"

        return render_template("index.html", prediction=result, img_path=filepath)

    return "No file uploaded"

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
