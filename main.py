from pathlib import Path

import yaml

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
        extract_checkerboard_frames(
            video, ROOT / "out", camera_id=camera_id,
            board_squares=config.get("board_squares", [9, 6]),
            frame_count=config.get("frame_count", 20),
            min_sharpness=config.get("min_sharpness", 100.0),
        )


if __name__ == "__main__":
    main()
