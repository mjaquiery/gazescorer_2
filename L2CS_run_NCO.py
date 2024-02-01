import os.path
import time

import cv2
import pandas
from l2cs import Pipeline, render
import torch
import pathlib

"""
Use the L2CS to predict the gaze direction in video files in the Number Comparison Online dataset.

Each video is processed frame by frame. Each frame is processed by the model, and the results are saved to a CSV file.
"""

save_videos = True
max_videos = 0

input_dir = os.path.join(r"./video_input/Number_comparison_online_new")
output_dir = os.path.join(r".output/L2CS_NCO_new")
gaze_pipeline = Pipeline(
    weights=pathlib.Path('models/Gaze360/L2CSNet_gaze360.pkl'),
    arch='ResNet50',
    device=torch.device(0)  # or 'gpu'
)

start_time = time.time()

results = pandas.DataFrame(columns=["video", "frame", "face_count", "pitch", "yaw"])

# Iterate over each video
video_file_count = 0
for video_file in os.listdir(input_dir):
    if 0 < max_videos <= video_file_count:
        break
    video_file_count += 1
    video_file_time = time.time()

    # Load the video and split into frames
    video = cv2.VideoCapture(os.path.join(input_dir, video_file))
    frame_number = 0
    frames = []
    errors = []
    while video.isOpened():
        face_count_warning = True

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
                if face_count_warning:
                    print(f"WARNING: Found {face_count} faces in frame {frame_number} of {video_file}. Using the first entry.")
                    face_count_warning = False

            yaw = yaw[0]
            pitch = pitch[0]

            results.loc[len(results)] = [
                video_file,
                frame_number,
                face_count,
                float(pitch),
                float(yaw),
            ]

            if save_videos:
                frames.append(render(frame, frame_results))
        except Exception as e:
            errors.append(e)
        frame_number += 1
    video.release()

    if save_videos and len(frames) > 0:
        os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
        video_out = cv2.VideoWriter(
            os.path.join(output_dir, "videos", f"{video_file}.mp4"),
            0,
            30,
            (frames[0].shape[1], frames[0].shape[0])
        )
        for frame in frames:
            video_out.write(frame)
        video_out.release()
    elif save_videos:
        print(f"WARNING: No frames found for {video_file}")

    if len(errors):
        print(f"Processed {video_file} in {round(time.time() - video_file_time, 2)}s with {len(errors)} errors: {errors}")
    else:
        print(f"Processed {video_file} in {round(time.time() - video_file_time, 2)}s")

fname = os.path.join(output_dir, "results.csv")
os.makedirs(os.path.dirname(fname), exist_ok=True)
results.to_csv(fname, index=False)

print(f"Finished processing. Time elapsed: {round(time.time() - start_time, 2)}s")

pass
