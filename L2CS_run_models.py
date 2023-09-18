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

input_dir_base = os.path.join(r"./video_input")
output_dir_base = os.path.join(r".output/L2CS_video_output")
labels = ["left", "right"]
gaze_pipeline = Pipeline(
    weights=pathlib.Path('models/Gaze360/L2CSNet_gaze360.pkl'),
    arch='ResNet50',
    device=torch.device(0)  # or 'gpu'
)

start_time = time.time()

results = pandas.DataFrame(columns=["video", "frame", "pitch", "yaw", "predicted_class", "actual_class"])
output_dir = os.path.join(r"./.output/L2CS_video_coding")

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
            frame_results = gaze_pipeline.step(frame)
            yaw = frame_results.yaw
            pitch = frame_results.pitch
            if len(yaw) != 1:
                print(f"WARNING: Found {len(yaw)} faces in frame {frame_number} of {video_file}. Using the first entry.")
                print(yaw)
                yaw = yaw[0]

            if len(pitch) != 1:
                print(f"WARNING: Found {len(pitch)} faces in frame {frame_number} of {video_file}. Using the first entry.")
                print(pitch)
                pitch = pitch[0]

            results.loc[len(results)] = [
                video_file,
                frame_number,
                pitch,
                yaw,
                labels[0 if yaw > 0 else 1],
                label
            ]
            frame_number += 1
            cv2.imwrite(
                os.path.join(output_dir, label, f"{video_file}_{frame_number}.png"),
                render(frame, frame_results)
            )
        video.release()

    print(f"Processed {video_file_count} videos in input {input_dir}. Time elapsed: {time.time() - label_time} seconds")

fname = os.path.join(output_dir, "results.csv")
os.makedirs(os.path.dirname(fname), exist_ok=True)
results.to_csv(fname, index=False)

print(f"Finished processing. Time elapsed: {time.time() - start_time} seconds")

pass