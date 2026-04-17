import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

# -----------------------------
# 1. Load Trained Model
# -----------------------------
model = keras.models.load_model("dog_cat_cnn_model.h5")

# -----------------------------
# 2. Load and Preprocess Image
# -----------------------------
img_path = "test_image.jpg"   # <-- change this

img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)

# Normalize (same as training)
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# -----------------------------
# 3. Predict
# -----------------------------
prediction = model.predict(img_array)

# -----------------------------
# 4. Output Result
# -----------------------------
if prediction[0][0] > 0.5:
    print("🐶 Dog")
else:
    print("🐱 Cat")

print("Confidence:", prediction[0][0])
