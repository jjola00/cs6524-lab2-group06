
import cv2
import numpy as np
from os import path as opath

PATTERN = (8, 5)    # inner corners of our board (9x6 squares)
SQUARE_CM = 2.5


def find_corners(video_path, pattern=PATTERN, num_frames=20):
    """PLACEHOLDER for Step 2 (teammate replaces this). Returns (image_points, image_size)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    image_points = []
    for index in np.linspace(0, total - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            image_points.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
    cap.release()
    return image_points, image_size


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
