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

save_videos = False

input_dir_base = os.path.join(r"./video_input/BRM_input/input/raw_videos")
output_dir_base = os.path.join(r".output/L2CS_gorilla")
input_dirs = ["adult/processed", "child/processed"]
gaze_pipeline = Pipeline(
    weights=pathlib.Path('models/Gaze360/L2CSNet_gaze360.pkl'),
    arch='ResNet50',
    device=torch.device(0)  # or 'gpu'
)

start_time = time.time()

results = pandas.DataFrame(columns=["video", "frame", "face_count", "pitch", "yaw", "predicted_class", "dir"])
output_dir = os.path.join(r"./.output/L2CS_video_coding")

for in_dir in input_dirs:
    label_time = time.time()
    input_dir = os.path.join(input_dir_base, in_dir)

    # Iterate over each video
    video_file_count = 0
    for video_dir in os.listdir(input_dir):
        for video_file in os.listdir(os.path.join(input_dir, video_dir)):
            video_file_count += 1
            video_file_time = time.time()

            # Load the video and split into frames
            video = cv2.VideoCapture(os.path.join(input_dir, video_dir, video_file))
            frame_number = 0
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                # Process the frame to extract eyes and save to numpy array
                try:
                    frame_results = gaze_pipeline.step(frame)
                    yaw = frame_results.yaw
                    pitch = frame_results.pitch
                    face_count = 1
                    if len(yaw) != 1 or len(pitch) != 1:
                        face_count = max(len(yaw), len(pitch))
                        print(f"WARNING: Found {face_count} faces in frame {frame_number} of {video_file}. Using the first entry.")
                        yaw = yaw[0]
                        pitch = pitch[0]

                    results.loc[len(results)] = [
                        video_file,
                        frame_number,
                        face_count,
                        float(pitch),
                        float(yaw),
                        'left' if pitch < 0 else 'right',
                        in_dir.split('/')[0]
                    ]
                    if save_videos:
                        os.makedirs(os.path.join(output_dir, in_dir.split('/')[0]), exist_ok=True)
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
                        cv2.imwrite(os.path.join(output_dir, in_dir.split('/')[0], f"{video_file}_{frame_number}.png"), img)
                except Exception as e:
                    print(f"ERROR: Failed to process frame {frame_number} of {video_file}: {e}")
                frame_number += 1
            video.release()

    print(f"Processed {video_file_count} videos in input {input_dir}. Time elapsed: {time.time() - label_time} seconds")

fname = os.path.join(output_dir, "results.csv")
os.makedirs(os.path.dirname(fname), exist_ok=True)
results.to_csv(fname, index=False)

print(f"Finished processing. Time elapsed: {time.time() - start_time} seconds")

pass
