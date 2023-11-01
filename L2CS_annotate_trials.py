import os.path
import time

import cv2
import pandas
from l2cs import Pipeline, render
import torch
import pathlib

"""
Use the L2CS to predict the gaze direction in video files.

Output will be videos with the gaze direction and eyetracker + L2CS scores drawn on each frame.
"""

n_videos = 1

input_dir = os.path.join(r"./video_input/trial")
output_dir = os.path.join(r".output/L2CS_annotated_trials")
gaze_pipeline = Pipeline(
    weights=pathlib.Path('models/Gaze360/L2CSNet_gaze360.pkl'),
    arch='ResNet50',
    device=torch.device(0)  # or 0 or 'gpu'
)

start_time = time.time()

et_data = pandas.read_csv(os.path.join(".input", "corrected_ET_output_results.csv"))

et_colour = (255, 0, 0)
l2cs_colour = (0, 0, 255)

# Iterate over each video
video_file_count = 0
for video_dir in os.listdir(input_dir):
    os.makedirs(os.path.join(output_dir, video_dir), exist_ok=True)
    for video_file in os.listdir(os.path.join(input_dir, video_dir)):
        if video_file_count >= n_videos:
            break
        video_file_count += 1
        video_file_time = time.time()

        video_name = video_file.split(".")[0]
        # Load the video and split into frames
        video = cv2.VideoCapture(os.path.join(input_dir, video_dir, video_file))
        frames = []
        while video.isOpened():
            frame_number = len(frames)
            video_multi_warning = True
            video_video_warning = True
            ret, frame = video.read()
            if not ret:
                break
            # Process the frame to extract eyes and save to numpy array
            try:
                frame_results = gaze_pipeline.step(frame)
                pitch = frame_results.pitch[0]
                face_count = 1

                try:
                    et_frame = et_data.loc[(et_data["video"] == video_file) & (et_data["frame"] == frame_number)]
                    if et_frame["status"].iloc[0] != 0:
                        et_value = -100
                        et_text = ""
                    else:
                        et_value = float(et_frame["left_gaze_x"].iloc[0])
                        et_text = round(et_value, 2)
                except Exception as e:
                    if video_video_warning:
                        print(f"WARNING: Could not find video {video_file} in ET data.")
                        video_video_warning = False
                    et_value = -100
                    et_text = ""

                pitch = round(float(pitch), 2)
                img = frame.copy()
                l2cs_text = f"L2CS {pitch} "
                cv2.putText(
                    img,
                    l2cs_text,
                    (25, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    l2cs_colour,
                    thickness=2
                )
                if et_text != "":
                    text_size, _ = cv2.getTextSize(l2cs_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.putText(
                        img,
                        f"ET {et_text}",
                        (25 + text_size[0], 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        et_colour,
                        thickness=2
                    )

                if -1 < pitch < 1:
                    cv2.circle(
                        center=(int(img.shape[1] * (1 - (pitch + 1) / 2)), int(img.shape[0] / 2)),
                        radius=5,
                        color=l2cs_colour,
                        thickness=-1,
                        img=img
                    )
                if -1 < et_value < 1:
                    cv2.circle(
                        center=(int(img.shape[1] * (1 - (et_value + 1) / 2)), int(img.shape[0] / 2)),
                        radius=5,
                        color=et_colour,
                        thickness=-1,
                        img=img
                    )
                img = render(img, frame_results)
                frames.append(img)
            except Exception as e:
                print(f"ERROR: Failed to process frame {frame_number} of {video_file}: {e}")
            frame_number += 1
        video.release()

        # Build pngs into video
        height, width, layers = frames[0].shape
        video = cv2.VideoWriter(os.path.join(output_dir, video_dir, video_file), 0, 30, (width, height))

        for img in frames:
            video.write(img)

        cv2.destroyAllWindows()
        video.release()
        print(f"Finished processing {video_file} in {time.time() - video_file_time} seconds")

print(f"Finished processing. Time elapsed: {time.time() - start_time} seconds")

pass
