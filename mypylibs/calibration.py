
import cv2
import numpy as np
from os import path as opath

from mypylibs.checkerboard import find_good_frames, select_frames

PATTERN = (8, 5)    # inner corners of our board (9x6 squares)
SQUARE_CM = 2.5


def find_corners(video_path, pattern=PATTERN, num_frames=20):
    """Step 2: return corners from sharp frames spread across the video."""
    good_frames, video_info = find_good_frames(video_path, pattern, min_sharpness=100.0)
    selected = select_frames(good_frames, num_frames)
    image_points = [frame["corners"] for frame in selected]
    return image_points, tuple(video_info["image_size"])


def calibrate(image_points, image_size, pattern=PATTERN, square_cm=SQUARE_CM):
    """Step 3: returns K, D and the mean reprojection error in pixels."""
    # Real-world corners (i*s, j*s, 0) in cm
    board = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_cm
    object_points = [board] * len(image_points)

    _, K, D, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

    # Mean distance between detected corners and where K, D predict them
    errors = []
    for i in range(len(image_points)):
        projected, _ = cv2.projectPoints(board, rvecs[i], tvecs[i], K, D)
        errors.append(np.linalg.norm(image_points[i].reshape(-1, 2) - projected.reshape(-1, 2), axis=1).mean())

    return K, D.ravel(), np.mean(errors)


def save(K, D, camera_id, root_out='out'):
    np.savetxt(opath.join(root_out, f'{camera_id}_K.csv'), K, delimiter=',', fmt='%.4f')
    np.savetxt(opath.join(root_out, f'{camera_id}_D.csv'), D, delimiter=',', fmt='%.6f')
