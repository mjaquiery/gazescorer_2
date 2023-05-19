import re

import numpy as np
import random
import os
import cv2
import math
import time

import pandas

"""
Produce test and training datasets from the Columbia Gaze Data Set.

Eye images are converted to small greyscale images and then further processed to numpy arrays.
"""

for strategy in ["diff", "default", "diff_threshold", "default_threshold"]:

    random.seed(20230512)

    input_dir = r".output/images/eyes"
    output_dir = os.path.join(r".output/stimuli", strategy, "eyes")

    # Extract face from images and save to output_dir
    img_count = math.inf
    output_count = 0
    files_processed = 0

    start_time = time.time()

    resolution = 28  # images will be in resolution x resolution greyscale

    img_arr = np.array([], dtype=np.float32).reshape((0, resolution, resolution))
    labels = []
    metadata = pandas.DataFrame(columns=["file", "magnitude"])

    # Walk through the root directory and assign each file to a new directory based on regex matches
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if img_count <= 0:
                break
            if os.path.basename(root) == "neutral":
                continue
            files_processed += 1
            save_path = os.path.join(output_dir, file)
            img = cv2.imread(os.path.join(root, file))
            if img is None:
                print(f"Could not read image {os.path.join(root, file)}")
                continue
            img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_lowres = cv2.resize(img_greyscale, (resolution, resolution))
            if re.search(r"_threshold", strategy):
                # Otsu's thresholding
                _, img_lowres = cv2.threshold(img_lowres, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if strategy == "diff":
                # Load the neutral version of this image
                try:
                    match = re.search(r"(.+)_-?\d+H(.+)", file)
                    neutral_img = cv2.imread(os.path.join(input_dir, 'neutral', f"{match.group(1)}_0H{match.group(2)}"))
                    neutral_greyscale = cv2.cvtColor(neutral_img, cv2.COLOR_BGR2GRAY)
                    neutral_lowres = cv2.resize(neutral_greyscale, (resolution, resolution))
                    if re.search(r"_threshold", strategy):
                        # Otsu's thresholding
                        _, neutral_lowres = cv2.threshold(neutral_lowres, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    img_lowres = cv2.subtract(img_lowres, neutral_lowres)
                except:
                    print(f"Could not find neutral version of {file}")
                    continue

            # Convert the images to numpy arrays
            img_arr = np.append(img_arr, img_lowres)
            labels.append(os.path.basename(root))
            magnitude = re.search(r"_(-?\d+)H", file).group(1)
            metadata.loc[len(metadata)] = [file, magnitude]

            # Save the images
            if os.path.exists(save_path):
                continue
            cv2.imwrite(save_path, img_lowres)
            output_count += 1
            img_count -= 1

    img_arr = img_arr.reshape((-1, resolution, resolution))

    assert img_arr.shape[0] == len(labels)

    os.makedirs(os.path.join(r"./.output/numpy", strategy), exist_ok=True)
    os.makedirs(os.path.join(r"./.output/models", strategy), exist_ok=True)
    np.save(os.path.join(r"./.output/numpy", strategy, "images.npy"), img_arr)
    np.save(os.path.join(r"./.output/numpy", strategy, "labels.npy"), labels)
    metadata.to_csv(os.path.join(r"./.output/models", strategy, "metadata.csv"), index=False)

    print(f'Done! Processed {files_processed} images and saved {output_count} new stimuli to {output_dir} in {round(time.time() - start_time, 2)} seconds.')
