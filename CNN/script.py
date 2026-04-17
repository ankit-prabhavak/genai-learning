import os
import shutil
import random

source_dir = "PetImages"
train_dir = "dataset/training_set"
test_dir = "dataset/test_set"

classes = ['cats', 'dogs']

for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    files = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(files)

    split = int(0.8 * len(files))

    train_files = files[:split]
    test_files = files[split:]

    for f in train_files:
        shutil.copy(
            os.path.join(source_dir, cls, f),
            os.path.join(train_dir, cls, f)
        )

    for f in test_files:
        shutil.copy(
            os.path.join(source_dir, cls, f),
            os.path.join(test_dir, cls, f)
        )