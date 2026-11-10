"""Extract clear checkerboard frames and find their inner corners."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np


def find_good_frames(video_path, pattern, min_sharpness):
    """Read the video and keep sharp frames where all corners are found."""
    video = cv2.VideoCapture(str(video_path))
    good_frames = []
    frame_index = 0
    detected_count = 0
    image_size = None

    # Stop corner refinement after 40 attempts or when movement is tiny.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    try:
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        expected_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            success, frame = video.read()
            if not success:
                break

            size = (frame.shape[1], frame.shape[0])
            if image_size is not None and size != image_size:
                raise ValueError("Video changes image size")
            image_size = size

            # OpenCV detects corners in a greyscale image.
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(grey, pattern)

            if found:
                detected_count += 1
                corners = cv2.cornerSubPix(grey, corners, (5, 5), (-1, -1), criteria)
                # Keep the same array shape on Windows and in Docker.
                corners = corners.reshape(-1, 1, 2)

                # Measure sharpness around the board, rather than the background.
                x, y, width, height = cv2.boundingRect(corners)
                board = grey[y:y + height, x:x + width]
                sharpness = float(cv2.Laplacian(board, cv2.CV_64F).var())

                if sharpness >= min_sharpness:
                    good_frames.append({
                        "index": frame_index,
                        "corners": corners,
                        "sharpness": sharpness,
                    })

            frame_index += 1

        if expected_frames > 0 and frame_index < expected_frames:
            raise RuntimeError(f"Decoded only {frame_index}/{expected_frames} video frames")
    finally:
        video.release()

    if not np.isfinite(fps) or fps <= 0:
        fps = None
    duration = frame_index / fps if fps else None
    video_info = {
        "fps": fps,
        "duration_seconds": duration,
        "image_size": image_size,
        "decoded_frames": frame_index,
        "detected_frames": detected_count,
    }
    return good_frames, video_info


def select_frames(good_frames, frame_count):
    """Split the usable frames into groups and take the sharpest in each."""
    if len(good_frames) < frame_count:
        raise ValueError(
            f"Need {frame_count} detectable, sharp frames; found {len(good_frames)}. "
            "Check the board settings and recording."
        )

    selected_frames = []
    for group in range(frame_count):
        start = group * len(good_frames) // frame_count
        end = (group + 1) * len(good_frames) // frame_count
        frames_in_group = good_frames[start:end]
        sharpest = frames_in_group[0]
        for frame in frames_in_group:
            if frame["sharpness"] > sharpest["sharpness"]:
                sharpest = frame
        selected_frames.append(sharpest)

    return selected_frames


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot save {path}")


def save_contact_sheet(images, selected_frames, path):
    """Put small corner overlays together so the chosen views are easy to check."""
    columns = 5
    cell_width = 240
    cell_height = 160
    rows = (len(images) + columns - 1) // columns
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 255, dtype=np.uint8)

    for index, image in enumerate(images):
        scale = min(cell_width / image.shape[1], 135 / image.shape[0])
        width = round(image.shape[1] * scale)
        height = round(image.shape[0] * scale)
        thumbnail = cv2.resize(image, (width, height))

        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet[y + 25:y + 25 + height, x:x + width] = thumbnail

        source_frame = selected_frames[index]["index"]
        label = f"{index + 1:02d}: source {source_frame}"
        cv2.putText(sheet, label, (x + 3, y + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)

    save_image(path, sheet)


def save_frames(video_path, destination, selected_frames, pattern):
    """Save the selected images, corner coordinates and a list of source frames."""
    for folder in ("frames", "grayscale", "corners"):
        (destination / folder).mkdir(parents=True, exist_ok=True)

    overlays = []
    video = cv2.VideoCapture(str(video_path))
    try:
        for number, selected in enumerate(selected_frames, start=1):
            video.set(cv2.CAP_PROP_POS_FRAMES, selected["index"])
            success, frame = video.read()
            if not success:
                raise RuntimeError(f"Cannot reread video frame {selected['index']}")

            filename = f"frame_{number:02d}.png"
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            save_image(destination / "frames" / filename, frame)
            save_image(destination / "grayscale" / filename, grey)

            cv2.drawChessboardCorners(frame, pattern, selected["corners"], True)
            save_image(destination / "corners" / filename, frame)
            overlays.append(frame)
    finally:
        video.release()

    save_contact_sheet(overlays, selected_frames, destination / "contact_sheet.jpg")
    corners = np.array([frame["corners"] for frame in selected_frames])
    np.save(destination / "corners.npy", corners)

    with (destination / "selected_frames.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["saved_frame", "source_frame_index_zero_based", "sharpness"])
        for number, frame in enumerate(selected_frames, start=1):
            writer.writerow([f"frame_{number:02d}.png", frame["index"], frame["sharpness"]])


def extract_checkerboard_frames(
    video_path, output_dir, camera_id="01", board_squares=(9, 6),
    frame_count=20, min_sharpness=100.0,
):
    """Run steps 1 and 2: check the video, find corners, select frames and save."""
    if len(board_squares) != 2 or any(type(size) is not int or size < 3 for size in board_squares):
        raise ValueError("board_squares must contain two integer square counts >= 3")
    if type(frame_count) is not int or frame_count < 20:
        raise ValueError("Use at least 20 calibration frames")
    if not np.isfinite(min_sharpness) or min_sharpness < 0:
        raise ValueError("min_sharpness must be a non-negative finite number")

    video_path = Path(video_path)
    size_bytes = video_path.stat().st_size
    if size_bytes > 5_000_000:
        raise ValueError(
            f"{video_path.name} exceeds 5 MB. Reduce its size and apply the "
            "same image preprocessing to future scene recordings."
        )

    # A 9 x 6 square board has 8 x 5 inner corners.
    columns, rows = board_squares
    pattern = (columns - 1, rows - 1)
    print(f"Camera {camera_id}: {video_path.name}, {size_bytes / 1_000_000:.2f} MB")

    good_frames, video_info = find_good_frames(video_path, pattern, min_sharpness)
    duration = video_info["duration_seconds"]
    if duration is None or not 5 <= duration <= 10:
        print("Warning: the recommended recording duration is 5 to 10 seconds.")

    selected_frames = select_frames(good_frames, frame_count)
    destination = Path(output_dir) / f"{camera_id}_checkerboard"
    save_frames(video_path, destination, selected_frames, pattern)

    summary = {
        "video": video_path.name,
        "file_size_bytes": size_bytes,
        "opencv_version": cv2.__version__,
        "board_squares": list(board_squares),
        "inner_corners": list(pattern),
        "sharp_candidate_frames": len(good_frames),
        "min_sharpness": min_sharpness,
        "selected_frame_indices_zero_based": [frame["index"] for frame in selected_frames],
        "selection": "Sharpest frame from each group of usable frames across the video",
        "preprocessing": {"resize": None, "crop": None, "rotation": None},
    }
    summary.update(video_info)
    with (destination / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")

    print(f"Checkerboard detected in {video_info['detected_frames']} frames.")
    print(f"Saved {len(selected_frames)} frames, greyscale images and corner overlays in {destination}")
    return summary
