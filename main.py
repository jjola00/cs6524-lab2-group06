
from pathlib import Path

import yaml
import numpy as np

from mypylibs import calibration
from mypylibs import camera_pose
from mypylibs.checkerboard import extract_checkerboard_frames


ROOT = Path(__file__).resolve().parent


def main():

    input_dir = ROOT / "in"
    output_dir = ROOT / "out"

    # --------------------------------------------
    # STEP 3: CAMERA CALIBRATION
    # --------------------------------------------

    camera_id = "03"

    calibration_video = (
        input_dir / "03_caliberation.mp4"
    )

    if not calibration_video.is_file():
        raise FileNotFoundError(calibration_video)

    config_path = (
        input_dir / f"{camera_id}_calibration.yaml"
    )

    config = {}

    if config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("Invalid calibration configuration.")

    summary = extract_checkerboard_frames(
        calibration_video,
        output_dir,
        camera_id=camera_id,
        board_squares=config.get("board_squares", [9, 6]),
        frame_count=config.get("frame_count", 20),
        min_sharpness=config.get("min_sharpness", 100.0),
    )

    corners_path = (
        output_dir /
        f"{camera_id}_checkerboard" /
        "corners.npy"
    )

    image_points = list(np.load(corners_path))

    image_size = tuple(summary["image_size"])
    pattern = tuple(summary["inner_corners"])

    K, D, error = calibration.calibrate(
        image_points,
        image_size,
        pattern=pattern
    )

    calibration.save(
        K, D, camera_id, output_dir
    )

    np.set_printoptions(precision=4, suppress=True)

    print("\nSTEP 3: CAMERA CALIBRATION")
    print(f"K =\n{K}")
    print(f"D = {D}")
    print(f"Mean reprojection error = {error:.4f} px")

    if error > 0.5:
        print(
            "WARNING: Calibration error exceeds "
            "the assignment target of 0.5 px."
        )


    # --------------------------------------------
    # STEP 5: INTER-FRAME CAMERA POSE
    # --------------------------------------------

    scene_videos = [
        input_dir / "03_01.mp4",
        input_dir / "03_02.mp4"
    ]

    for video in scene_videos:

        if not video.is_file():
            print(f"Skipping missing video: {video}")
            continue

        print(f"\nSTEP 5: Processing {video.name}")

        pose_output = (
            output_dir /
            f"{camera_id}_pose" /
            video.stem
        )

        camera_pose.estimate_camera_pose(
            video_path=video,
            K=K,
            D=D,
            output_dir=pose_output,
            calibration_size=image_size,
            frame_step=1
        )



if __name__ == "__main__":
    main()
