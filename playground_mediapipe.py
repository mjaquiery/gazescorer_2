from __future__ import annotations

import numpy as np
import random
import os
import cv2
import math
import shutil
import re
import time
import mediapipe
import dlib
from imutils import face_utils
from retinaface import RetinaFace

"""
Produce test and training datasets from the Columbia Gaze Data Set.

Test and training datasets are produced by categorising the Columbia Gaze Data Set by gaze direction,
and different datasets are created for the whole face and for eye crops.
"""

random.seed(20230512)

def create_path(path: os.PathLike|str, **kwargs):
    """
    Create a path if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

input_dir = r".input/matt"
output_dir = r".output/matt"

# Prepare training and test data by categorising Columbia Gaze Data Set by gaze direction
out_dirs = {
    "any": [re.compile(r".*\.png")]
}

# Extract face from images and save to output_dir
img_count = math.inf
output_count = 0
files_processed = 0
errors = []

mp_face_detection = mediapipe.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.4)
predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
dlib_eyes = {
    "left_eye": [37, 38, 39, 40, 41, 42],
    "right_eye": [43, 44, 45, 46, 47, 48],
}

start_time = time.time()


# Walk through the root directory and assign each file to a new directory based on regex matches
for root, dirs, files in os.walk(input_dir):
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

            print(f"Processing {os.path.join(root, file)}")

            # Save the whole image
            try:
                shutil.copy(
                    os.path.join(root, file),
                    create_path(os.path.join(output_dir, 'images', 'faces', out_dir, file))
                )
            except shutil.SameFileError:
                pass
            output_count += 1

            # Process with mediapipe
            engine = 'mediapipe'
            # Save the eyes
            img = cv2.imread(os.path.join(root, file))
            faces = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not faces.detections:
                errors.append(f"{engine}: Could not find face in {os.path.join(root, file)}")
            else:
                face = faces.detections[0]
                confidence = face.score
                bounding_box = face.location_data.relative_bounding_box


                x = int(bounding_box.xmin * img.shape[1])
                w = int(bounding_box.width * img.shape[1])
                y = int(bounding_box.ymin * img.shape[0])
                h = int(bounding_box.height * img.shape[0])

                whole_img = img.copy()
                cv2.rectangle(whole_img, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
                cv2.putText(whole_img, f"{confidence[0]:.2f}", (x, y + 100), 3, 5, (255, 255, 255))
                cv2.imwrite(
                    create_path(os.path.join(output_dir, 'images', 'whole', f"{engine}_{out_dir}-whole_{file}")),
                    whole_img
                )

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), thickness=2)
                cv2.putText(img, f"{round(confidence[0], 2)}", (int(x + w/2), y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)

                landmarks = face.location_data.relative_keypoints

                right_eye = (int(landmarks[0].x * img.shape[1]), int(landmarks[0].y * img.shape[0]))
                left_eye = (int(landmarks[1].x * img.shape[1]), int(landmarks[1].y * img.shape[0]))
                nose = (int(landmarks[2].x * img.shape[1]), int(landmarks[2].y * img.shape[0]))
                mouth = (int(landmarks[3].x * img.shape[1]), int(landmarks[3].y * img.shape[0]))
                right_ear = (int(landmarks[4].x * img.shape[1]), int(landmarks[4].y * img.shape[0]))
                left_ear = (int(landmarks[5].x * img.shape[1]), int(landmarks[5].y * img.shape[0]))

                cv2.circle(img, right_eye, 5, (255, 0, 0), -1)
                cv2.circle(img, left_eye, 5, (0, 255, 0), -1)
                cv2.circle(img, nose, 5, (0, 0, 255), -1)
                cv2.circle(img, mouth, 5, (0, 0, 255), -1)
                cv2.circle(img, right_ear, 5, (255, 0, 255), -1)
                cv2.circle(img, left_ear, 5, (0, 255, 255), -1)

                cv2.imwrite(
                    create_path(os.path.join(output_dir, 'images', 'eyes', f"{engine}_{out_dir}-left_eye_{file}")),
                    img[left_eye[1] - 50:left_eye[1] + 50, left_eye[0] - 50:left_eye[0] + 50]
                )
                cv2.imwrite(
                    create_path(os.path.join(output_dir, 'images', 'eyes', f"{engine}_{out_dir}-right_eye_{file}")),
                    img[right_eye[1] - 50:right_eye[1] + 50, right_eye[0] - 50:right_eye[0] + 50]
                )
                cv2.imwrite(
                    create_path(os.path.join(output_dir, 'images', 'face', f"{engine}_{out_dir}-face_{file}")),
                    img
                )

            engine = 'dlib'
            img = cv2.imread(os.path.join(root, file))
            img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(img_greyscale, 1)
            if not len(rects) > 0:
                errors.append(f"{engine}: Could not find face in {os.path.join(root, file)}")
            else:
                # https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
                # loop over the face detections
                for (i, rect) in enumerate(rects):
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(img_greyscale, rect)
                    shape = face_utils.shape_to_np(shape)

                    face_img = img.copy()
                    [cv2.circle(face_img, p, 5, (255, 0, 0), -1) for p in shape]

                    cv2.imwrite(
                        create_path(os.path.join(output_dir, 'images', 'face', f"{engine}_{out_dir}-face_{file}")),
                        face_img
                    )
                    # convert dlib's rectangle to a OpenCV-style bounding box
                    # [i.e., (x, y, w, h)], then draw the face bounding box
                    # Extract eyes as square images
                    for eye, points in dlib_eyes.items():
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[p - 1] for p in points]))
                        if w < h:
                            x -= math.ceil((h - w) / 2)
                            w = h
                        elif h < w:
                            y -= math.ceil((w - h) / 2)
                            h = w
                        eye_img = img[y:y + h, x:x + w]
                        fname = create_path(os.path.join(output_dir, 'images', 'eyes', f"{engine}_{out_dir}-{eye}_{file}"))
                        cv2.imwrite(fname, eye_img)
                        output_count += 1

            engine = 'retinaface'
            img = cv2.imread(os.path.join(root, file))
            resp = RetinaFace.detect_faces(os.path.join(root, file))
            if not resp or 'face_1' not in resp:
                errors.append(f"{engine}: Could not find face in {os.path.join(root, file)}")
            else:
                confidence = resp['face_1']['score']
                x = int(resp['face_1']['facial_area'][0])
                y = int(resp['face_1']['facial_area'][1])
                x_end = int(resp['face_1']['facial_area'][2])
                y_end = int(resp['face_1']['facial_area'][3])
                cv2.rectangle(img, (x, y), (x_end, y_end), (0, 0, 0), thickness=2)
                cv2.putText(img, f"{round(confidence, 2)}", (int(x + (x_end - x)/2), y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)

                right_eye = [int(p) for p in resp['face_1']['landmarks']['right_eye']]
                left_eye = [int(p) for p in resp['face_1']['landmarks']['left_eye']]
                nose = [int(p) for p in resp['face_1']['landmarks']['nose']]
                mouth_right = [int(p) for p in resp['face_1']['landmarks']['mouth_right']]
                mouth_left = [int(p) for p in resp['face_1']['landmarks']['mouth_left']]

                cv2.circle(img, right_eye, 5, (255, 0, 0), -1)
                cv2.circle(img, left_eye, 5, (0, 255, 0), -1)
                cv2.circle(img, nose, 5, (0, 0, 255), -1)
                cv2.circle(img, mouth_right, 5, (0, 0, 255), -1)
                cv2.circle(img, mouth_left, 5, (0, 0, 255), -1)

                cv2.imwrite(
                    create_path(os.path.join(output_dir, 'images', 'eyes', f"{engine}_{out_dir}-left_eye_{file}")),
                    img[left_eye[1] - 50:left_eye[1] + 50, left_eye[0] - 50:left_eye[0] + 50]
                )
                cv2.imwrite(
                    create_path(os.path.join(output_dir, 'images', 'eyes', f"{engine}_{out_dir}-right_eye_{file}")),
                    img[right_eye[1] - 50:right_eye[1] + 50, right_eye[0] - 50:right_eye[0] + 50]
                )
                cv2.imwrite(
                    create_path(os.path.join(output_dir, 'images', 'face', f"{engine}_{out_dir}-face_{file}")),
                    img
                )

                engine = 'retinaface_then_dlib'
                try:
                    img = cv2.imread(os.path.join(root, file))[
                          int(y/2):int(y_end + (img.shape[0] - y_end)/2),
                          int(x/2):int(x_end + (img.shape[1] - x_end)/2)
                          ]
                    assert img.size > 0
                except AssertionError:
                    img = cv2.imread(os.path.join(root, file))[y:y_end, x:x_end]
                cv2.imwrite(
                    create_path(os.path.join(output_dir, 'images', 'face', f"{engine}_{out_dir}-raw_{file}")),
                    img
                )
                img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(img_greyscale, 1)
                if not len(rects) > 0:
                    errors.append(f"{engine}: Could not find face in {os.path.join(root, file)}")
                else:
                    # https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
                    # loop over the face detections
                    for (i, rect) in enumerate(rects):
                        # determine the facial landmarks for the face region, then
                        # convert the facial landmark (x, y)-coordinates to a NumPy
                        # array
                        shape = predictor(img_greyscale, rect)
                        shape = face_utils.shape_to_np(shape)

                        face_img = img.copy()
                        [cv2.circle(face_img, p, 5, (255, 0, 0), -1) for p in shape]

                        cv2.imwrite(
                            create_path(os.path.join(output_dir, 'images', 'face', f"{engine}_{out_dir}-face_{file}")),
                            face_img
                        )
                        # convert dlib's rectangle to a OpenCV-style bounding box
                        # [i.e., (x, y, w, h)], then draw the face bounding box
                        # Extract eyes as square images
                        for eye, points in dlib_eyes.items():
                            (x, y, w, h) = cv2.boundingRect(np.array([shape[p - 1] for p in points]))
                            if w < h:
                                x -= math.ceil((h - w) / 2)
                                w = h
                            elif h < w:
                                y -= math.ceil((w - h) / 2)
                                h = w
                            eye_img = img[y:y + h, x:x + w]
                            fname = create_path(
                                os.path.join(output_dir, 'images', 'eyes', f"{engine}_{out_dir}-{eye}_{file}"))
                            cv2.imwrite(fname, eye_img)
                            output_count += 1

            img_count -= 1

print(f'Done! Processed {files_processed} files and produced {output_count} new images in {round(time.time() - start_time, 2)} seconds.')
if errors:
    print(f'Encountered {len(errors)} errors:')
    for e in errors:
        print(e)
    print(f"Error count by engine:")
    for engine in {e.split(':')[0] for e in errors}:
        print(f"{engine}: {len([e for e in errors if engine in e])}")
else:
    print('No errors encountered.')

