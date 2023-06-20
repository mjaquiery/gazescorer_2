import math
import os.path
import re
import time

import cv2
import dlib
import pandas
import tensorflow as tf
import numpy as np
import random

from imutils import face_utils

"""
Use the models trained in train_models.py to predict the gaze direction in video files.
Videos are segmented into directories by label.

Each video is processed frame by frame. Each frame is processed by the model, and the results are saved to a CSV file.
"""

input_dir_base = os.path.join(r"./video_input")
labels = ["left", "right"]
predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")
eyes = {
    "left_eye": [37, 38, 39, 40, 41, 42],
    "right_eye": [43, 44, 45, 46, 47, 48],
}

detector = dlib.get_frontal_face_detector()
resolution = 28  # images will be in resolution x resolution greyscale

for strategy in [r"default_threshold", "default"]:# [r"diff", r"default", r"diff_threshold", r"default_threshold"]:
    start_time = time.time()
    random.seed(20230512)

    model = tf.keras.models.load_model(os.path.join(r"./.output/models", strategy, "model.h5"))
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    results = pandas.DataFrame(columns=["video", "frame", "eye", "prob_left", "predicted_class", "actual_class"])
    output_dir = os.path.join(r"./.output/video_coding", strategy)

    for label in labels:
        label_time = time.time()
        input_dir = os.path.join(input_dir_base, label)

        # Iterate over each video
        video_file_count = 0
        for video_file in os.listdir(input_dir):
            video_file_count += 1
            video_file_time = time.time()

            # Load the video and split into frames
            video = cv2.VideoCapture(os.path.join(input_dir, video_file))
            frame_number = 0
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                # Process the frame to extract eyes and save to numpy array
                img_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(img_greyscale, 1)
                # loop over the face detections
                for (i, rect) in enumerate(rects):
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(img_greyscale, rect)
                    shape = face_utils.shape_to_np(shape)
                    # convert dlib's rectangle to a OpenCV-style bounding box
                    # [i.e., (x, y, w, h)], then draw the face bounding box
                    # Extract eyes as square images
                    for eye, points in eyes.items():
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[p - 1] for p in points]))
                        if w < h:
                            x -= math.ceil((h - w) / 2)
                            w = h
                        elif h < w:
                            y -= math.ceil((w - h) / 2)
                            h = w
                        eye_crop = img_greyscale[y:y + h, x:x + w]
                        img_lowres = cv2.resize(img_greyscale, (resolution, resolution))
                        if re.search(r"_threshold", strategy):
                            # Otsu's thresholding
                            _, img_lowres = cv2.threshold(img_lowres, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        eye_img = img_lowres

                        # Test model fit on test data
                        test_images = np.array([eye_img])
                        test_labels = np.array([0 if label == "left" else 1])
                        predictions = probability_model.predict(test_images, verbose=0)
                        results.loc[len(results)] = [
                            video_file,
                            frame_number,
                            eye,
                            predictions[0][0],
                            labels[np.argmax(predictions[0])],
                            labels[test_labels[0]]
                        ]
                frame_number += 1
            video.release()

        print(f"Processed {video_file_count} videos in input {input_dir}. Time elapsed: {time.time() - label_time} seconds")

    fname = os.path.join(output_dir, "results.csv")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    results.to_csv(fname, index=False)

    print(f"Finished processing {strategy}. Time elapsed: {time.time() - start_time} seconds")

pass
