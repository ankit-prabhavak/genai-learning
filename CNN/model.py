import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras import layers, models # type: ignore

# -----------------------------
# 1. Image Data Generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values (0-255 → 0-1)
    shear_range=0.2,         # Random distortion
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True     # Flip images (augmentation)
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# Load dataset from directories
train_data = train_datagen.flow_from_directory(
    'dataset/training_set',     # folder path
    target_size=(128, 128),     # resize images
    batch_size=32,
    class_mode='binary'         # 2 classes → dog/cat
)

test_data = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# -----------------------------
# 2. Build CNN Model
# -----------------------------
model = models.Sequential()

# First Convolution + Pooling
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Second Convolution + Pooling
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Third Convolution + Pooling
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Flatten
model.add(layers.Flatten())

# Fully Connected Layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))   # prevent overfitting
model.add(layers.Dense(1, activation='sigmoid'))  # binary output

# -----------------------------
# 3. Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 4. Train Model
# -----------------------------
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# -----------------------------
# 5. Save Model
# -----------------------------
model.save("dog_cat_cnn_model.h5")

print("Model training completed and saved!")
