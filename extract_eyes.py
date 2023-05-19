import numpy as np
import random
import os
import dlib
import cv2
from imutils import face_utils
import math
import shutil
import re
import time

"""
Produce test and training datasets from the Columbia Gaze Data Set.

Test and training datasets are produced by categorising the Columbia Gaze Data Set by gaze direction,
and different datasets are created for the whole face and for eye crops.
"""

random.seed(20230512)

input_dirs = [r"columbia_gaze_data_set", r"columbia_gaze_data_set"]
output_dir = r".output/"

# Prepare training and test data by categorising Columbia Gaze Data Set by gaze direction
root_dir = r"columbia_gaze_data_set"
out_dirs = {
    "neutral": [re.compile(r"_0P_-?\d+V_0H")],
    "left": [re.compile(r"_0P_-?\d+V_-[1-9]\d*H")],
    "right": [re.compile(r"_0P_-?\d+V_[1-9]\d*H")],
}

predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")
eyes = {
    "left_eye": [37, 38, 39, 40, 41, 42],
    "right_eye": [43, 44, 45, 46, 47, 48],
}

detector = dlib.get_frontal_face_detector()

# Extract face from images and save to output_dir
img_count = math.inf
output_count = 0
files_processed = 0

start_time = time.time()

# Create the new directories
for out_dir in out_dirs.keys():
    os.makedirs(os.path.join(root_dir, out_dir), exist_ok=True)

# Walk through the root directory and assign each file to a new directory based on regex matches
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if img_count <= 0:
            break
        files_processed += 1
        for out_dir, regexes in out_dirs.items():
            process = False
            for regex in regexes:
                if regex.search(file):
                    process = True
            if not process:
                continue

            if os.path.exists(os.path.join(output_dir, 'images', 'faces', out_dir, file)) and \
                os.path.exists(os.path.join(output_dir, 'images', 'eyes', out_dir, f"left_eye_{file}")) and \
                os.path.exists(os.path.join(output_dir, 'images', 'eyes', out_dir, f"right_eye_{file}")):
                continue

            print(f"Processing {os.path.join(root, file)}")

            # Save the whole image
            os.makedirs(os.path.join(output_dir, 'images', 'faces', out_dir), exist_ok=True)
            shutil.copy(os.path.join(root, file), os.path.join(output_dir, 'images', 'faces', out_dir, file))
            output_count += 1

            # Save the eyes
            img = cv2.imread(os.path.join(root, file))
            img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(img_greyscale, 1)
            # https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
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
                    eye_img = img[y:y + h, x:x + w]
                    fname = os.path.join(output_dir, 'images', 'eyes', out_dir, f"{eye}_{file}")
                    os.makedirs(os.path.dirname(fname), exist_ok=True)
                    cv2.imwrite(fname, eye_img)
                    output_count += 1

            img_count -= 1

print(f'Done! Processed {files_processed} files and produced {output_count} new images in {round(time.time() - start_time, 2)} seconds.')
