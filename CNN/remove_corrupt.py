from PIL import Image
import os

def clean_dataset(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()  # verify image
            except:
                print("Removing corrupted file:", path)
                os.remove(path)

# Run for both datasets
clean_dataset("dataset/training_set")
clean_dataset("dataset/test_set")

print("Dataset cleaned!")