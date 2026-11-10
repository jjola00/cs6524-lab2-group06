import cv2
import numpy as np
import os

video_path = "D:\Block15\ComputerVision\Wk3\lab2\in\chess.mp4"
output_folder = "calibration_frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#Select 20 evenly spaced frame positions
frame_indices = np.linspace(
    0,
    total_frames - 1,
    20,
    dtype=int
)

for i, frame_index in enumerate(frame_indices):

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

    ret, frame = cap.read()

    if not ret:
        continue

    filename = f"frame_{i+1:02d}.jpg"

    cv2.imwrite(
        os.path.join(output_folder, filename),
        frame
    )

cap.release()

print("20 calibration frames extracted.")