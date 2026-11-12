
from pathlib import Path

import yaml
import numpy as np

from mypylibs import calibration
from mypylibs import camera_pose
from mypylibs import stereo_depth
from mypylibs.checkerboard import extract_checkerboard_frames


ROOT = Path(__file__).resolve().parent


def main():

    input_dir = ROOT / "in"

    for calibration_video in sorted(input_dir.glob("[0-9][0-9]_calibration.mp4")):
        run_camera(calibration_video.name[:2], calibration_video)


def run_camera(camera_id, calibration_video):

    input_dir = ROOT / "in"
    output_dir = ROOT / "out"

    # --------------------------------------------
    # STEP 3: CAMERA CALIBRATION
    # --------------------------------------------

    print(f"\n===== CAMERA {camera_id} =====")

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
        frame_count=config.get("frame_count", 30),
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

    scene_videos = sorted(input_dir.glob(f"{camera_id}_[0-9][0-9].mp4"))

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

        pose_rows = camera_pose.estimate_camera_pose(
            video_path=video,
            K=K,
            D=D,
            output_dir=pose_output,
            calibration_size=image_size,
            frame_step=1
        )

        scene_config_path = input_dir / f"{video.stem}.yaml"
        scene_config = {}
        if scene_config_path.is_file():
            with scene_config_path.open(encoding="utf-8") as handle:
                scene_config = yaml.safe_load(handle) or {}
        if not isinstance(scene_config, dict):
            raise ValueError(f"Invalid scene configuration: {scene_config_path}")
        stereo_depth.create_depth_video(
            video, K, D, image_size, pose_rows, pose_output, output_dir,
            baseline_cm=scene_config.get("baseline_cm"),
            max_depth_cm=scene_config.get("max_depth_cm", 500.0),
        )



if __name__ == "__main__":
    main()
