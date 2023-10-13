import os.path
import time

import cv2
import pandas
from l2cs import Pipeline, render
import torch
import pathlib

"""
Use the L2CS to predict the gaze direction in video files.
Videos are segmented into directories by label.

Each video is processed frame by frame. Each frame is processed by the model, and the results are saved to a CSV file.
"""

files = [
    "./video_input/BRM_input/input/raw_videos/adult/processed/PID_A13_video.webm",
    "./video_input/BRM_input/input/raw_videos/adult/processed/PID_A12_video.webm",
    "./video_input/BRM_input/input/raw_videos/child/processed/PID_C5_video.webm",
    "./video_input/BRM_input/input/raw_videos/child/processed/PID_C9_video.webm",
    "./video_input/BRM_input/input/raw_videos/adult/processed/PID_A1_video.webm",
    "./video_input/BRM_input/input/raw_videos/adult/processed/PID_A3_video.webm",
]
gaze_pipeline = Pipeline(
    weights=pathlib.Path('models/Gaze360/L2CSNet_gaze360.pkl'),
    arch='ResNet50',
    device=torch.device(0)  # or 'gpu'
)

start_time = time.time()

results = pandas.DataFrame(columns=["video", "frame", "face_count", "pitch", "yaw", "predicted_class", "dir"])
output_dir = os.path.join(r"./.output/L2CS_inspect_gorilla")
video_file_count = 0

for video_file in files:
    video_file_count += 1
    video_file_time = time.time()

    # Load the video and split into frames
    video = cv2.VideoCapture(video_file)
    frame_number = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        # Process the frame to extract eyes and save to numpy array
        try:
            frame_results = gaze_pipeline.step(frame)
            os.makedirs(os.path.join(output_dir), exist_ok=True)
            img = render(frame, frame_results)
            cv2.putText(
                img,
                f"pitch {round(float(results.pitch[0]), 2)}",
                (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                thickness=2
            )
            cv2.putText(
                img,
                f"yaw {round(float(results.yaw[0]), 2)}",
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                thickness=2
            )
            cv2.imwrite(os.path.join(output_dir, f"{os.path.basename(video_file)}_{frame_number}.png"), img)
        except Exception as e:
            print(f"ERROR: Failed to process frame {frame_number} of {video_file}: {e}")
        frame_number += 1
    video.release()

print(f"Processed {video_file_count} videos. Time elapsed: {time.time() - start_time} seconds")
