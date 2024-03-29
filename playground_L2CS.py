#https://github.com/Ahmednull/L2CS-Net

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

    cap = cv2.VideoCapture(0)
    n_frames = 20
    n_delay = 0.5
    for i in range(n_frames):
        _, frame = cap.read()
        # cv2.imwrite(f'tmp_{i}_pre.png', frame)

        # Process frame and visualize
        results = gaze_pipeline.step(frame)
        frame = render(frame, results)
        cv2.putText(
            frame,
            f"pitch {round(float(results.pitch[0]), 2)}",
            (25, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            thickness=2
        )
        cv2.putText(
            frame,
            f"yaw {round(float(results.yaw[0]), 2)}",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            thickness=2
        )
        cv2.imwrite(f'L2CS_results/{i}.png', frame)
        if i < n_frames - 1:
            time.sleep(n_delay)

    pass
