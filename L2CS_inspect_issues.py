#https://github.com/Ahmednull/L2CS-Net
import os
import pathlib
import time

import torch
from l2cs import Pipeline, render
import cv2

if __name__ == '__main__':
    gaze_pipeline = Pipeline(
        weights=pathlib.Path('models/Gaze360/L2CSNet_gaze360.pkl'),
        arch='ResNet50',
        device=torch.device('cpu')  # or 'gpu'
    )
    target_files = [
        "video_input/left/0601/0601_trial_1.0_left_1.avi",
        "video_input/trial/1601/1601_trial_3.0.avi",
        "video_input/trial/0601/0601_trial_1.0.avi",
        "video_input/trial/0601/0601_trial_2.0.avi",
        "video_input/trial/2003/2003_trial_2.0.avi",
        "video_input/trial/2003/2003_trial_3.0.avi",
        "video_input/trial/2003/2003_trial_6.0.avi",
        "video_input/trial/2003/2003_trial_7.0.avi",
        "video_input/trial/2003/2003_trial_9.0.avi",
        "video_input/trial/2003/2003_trial_11.0.avi",
        "video_input/trial/2003/2003_trial_12.0.avi",
    ]

    start_time = time.time()

    output_dir = os.path.join(r"./.output/L2CS_inspection")

    for video_file in target_files:
        label_time = time.time()

        # Iterate over each video
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
                yaw = frame_results.yaw
                pitch = frame_results.pitch
                face_count = 1
                if len(yaw) != 1 or len(pitch) != 1:
                    face_count = max(len(yaw), len(pitch))
                    print(
                        f"WARNING: Found {face_count} faces in frame {frame_number} of {video_file}. Using the first entry.")
                    yaw = yaw[0]
                    pitch = pitch[0]

                os.makedirs(output_dir, exist_ok=True)
                img = render(frame, frame_results)
                cv2.imwrite(os.path.join(output_dir, f"{os.path.basename(video_file)}_{frame_number}.png"), img)
            except Exception as e:
                print(f"ERROR: Failed to process frame {frame_number} of {video_file}: {e}")
            frame_number += 1
        video.release()

        print(
            f"Processed {video_file}. Time elapsed: {time.time() - label_time} seconds")

    print(f"Finished processing. Time elapsed: {time.time() - start_time} seconds")

    pass
