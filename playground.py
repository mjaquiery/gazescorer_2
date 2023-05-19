from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import pandas as pd
import os
import re

# adapted from https://teachablemachine.withgoogle.com/train/image

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(r".output/keras_Model.h5", compile=False)

# Load the labels
class_names = open(r".output/labels.txt", "r").readlines()

# Compare predictions for each item in the .output/test/*_eye directory
dirs = {
    "left": r".output/test/left_eye",
    "right": r".output/test/right_eye"
}

results = pd.DataFrame(columns=['filename', 'eye', 'class_prediction', 'confidence_score', 'class', 'magnitude'])

for eye, dir in dirs.items():
    for root, dirs, files in os.walk(dir):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            img_small = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            img_arr = np.asarray(img_small, dtype=np.float32).reshape(1, 224, 224, 3)

            # Predicts the model
            prediction = model.predict(img_arr)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            magnitude = int(re.search(r"_(-?[0-9]+)H", file).group(1))
            actual_class = "left" if magnitude < 0 else "right"
            results.loc[len(results)] = [file, eye, class_name, confidence_score, actual_class, magnitude]

results.to_csv(r".output/results.csv", index=False)
print("Complete.")
