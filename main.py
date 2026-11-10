from pathlib import Path

import yaml
import numpy as np

from mypylibs import calibration
from mypylibs.checkerboard import extract_checkerboard_frames


ROOT = Path(__file__).resolve().parent


def main():
    input_dir = ROOT / "in"
    # Camera videos use names such as 01_calibration.mp4 and 02_calibration.mp4.
    videos = sorted(input_dir.glob("[0-9][0-9]_calibration.mp4"))
    # Keep the team's existing recording usable without renaming it.
    if (input_dir / "chess.mp4").is_file() and not (input_dir / "01_calibration.mp4").is_file():
        videos.insert(0, input_dir / "chess.mp4")
    if not videos:
        raise FileNotFoundError(f"No calibration videos found in {input_dir}")

    for video in videos:
        if video.name == "chess.mp4":
            camera_id = "01"
        else:
            camera_id = video.name[:2]

        # Read this camera's settings, or use the defaults below.
        config_path = input_dir / f"{camera_id}_calibration.yaml"
        config = {}
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Expected a settings mapping in {config_path}")
        # Find usable frames, select 20, and save the images and corners.
        summary = extract_checkerboard_frames(
            video, ROOT / "out", camera_id=camera_id,
            board_squares=config.get("board_squares", [9, 6]),
            frame_count=config.get("frame_count", 20),
            min_sharpness=config.get("min_sharpness", 100.0),
        )

        # Use these same 20 selected frames for the camera calibration.
        corners_path = ROOT / "out" / f"{camera_id}_checkerboard" / "corners.npy"
        image_points = list(np.load(corners_path))
        image_size = tuple(summary["image_size"])
        pattern = tuple(summary["inner_corners"])
        K, D, error = calibration.calibrate(image_points, image_size, pattern=pattern)
        calibration.save(K, D, camera_id, ROOT / "out")

        np.set_printoptions(precision=4, suppress=True)
        print(f"K =\n{K}\nD (k1, k2, p1, p2, k3) = {D}")
        print(f"Mean reprojection error = {error:.4f} px")


if __name__ == "__main__":
    main()
