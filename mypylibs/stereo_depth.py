"""Step 6: dense stereo. Unit translation needs an explicit physical scale.

Without baseline_cm, an assumed 1 cm per pair is clearly labelled throughout.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from mypylibs.camera_pose import _prepare_frame


def _matcher(minimum, number):
    return cv2.StereoSGBM_create(
        minDisparity=minimum, numDisparities=number, blockSize=5,
        P1=8*25, P2=32*25, disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2, preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def calculate_depth(left, right, K, D, R, t, baseline_cm=1.0, max_depth_cm=500.0):
    """Rectify and return left image, signed disparity, depth (NaN invalid), Q.

    Vertical stereo is transposed for matching then restored for reprojection.
    Both disparity signs are supported; right-left consistency rejects errors.
    """
    for name, value in (("baseline_cm", baseline_cm), ("max_depth_cm", max_depth_cm)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Stereo frames must be equally sized greyscale images")
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    norm = np.linalg.norm(t)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Cannot rectify zero or non-finite translation")
    t = t * (baseline_cm / norm)
    height, width = left.shape
    size = (width, height)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, D, K, D, size, R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    if not all(np.isfinite(m).all() for m in (R1, R2, P1, P2, Q)) or P1[0, 0] <= 0:
        raise ValueError("Invalid rectification geometry")
    maps = [cv2.initUndistortRectifyMap(K, D, r, p, size, cv2.CV_32FC1)
            for r, p in ((R1, P1), (R2, P2))]
    images = [cv2.remap(frame, *mapping, cv2.INTER_LINEAR)
              for frame, mapping in zip((left, right), maps)]
    masks = [cv2.erode(cv2.remap(np.full(left.shape, 255, np.uint8), *mapping,
                                cv2.INTER_NEAREST), np.ones((5, 5), np.uint8)) > 0
             for mapping in maps]
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])
    a, b = [np.ascontiguousarray(im.T if vertical else im) for im in images]
    ma, mb = [mask.T if vertical else mask for mask in masks]
    number = min(128, ((a.shape[1] // 3) // 16) * 16)
    if number < 16:
        raise ValueError("Frames too small for stereo matching")
    minimum = 0 if P2[1 if vertical else 0, 3] < 0 else -number
    reverse_minimum = -minimum - number + 1
    disparity = _matcher(minimum, number).compute(a, b).astype(np.float32) / 16.0
    reverse = _matcher(reverse_minimum, number).compute(b, a).astype(np.float32) / 16.0
    yy, xx = np.indices(a.shape)
    target_x = np.rint(xx - disparity).astype(np.int32)
    inside = (target_x >= 0) & (target_x < a.shape[1])
    target_x = np.clip(target_x, 0, a.shape[1] - 1)
    other = reverse[yy, target_x]
    valid = (inside & ma & mb[yy, target_x]
             & (disparity > minimum - 1) & (other > reverse_minimum - 1)
             & (np.abs(disparity) >= 0.5) & (np.abs(disparity + other) <= 1.5))
    if vertical:
        disparity, valid = disparity.T.copy(), valid.T.copy()
    depth = cv2.reprojectImageTo3D(disparity, Q)[:, :, 2]
    valid &= np.isfinite(depth) & (depth > 0) & (depth <= max_depth_cm)
    return (images[0], np.where(valid, disparity, np.nan).astype(np.float32),
            np.where(valid, depth, np.nan).astype(np.float32), Q)


class _DepthPlot:
    """Headless Matplotlib heatmap, with a fixed scale throughout each video."""
    def __init__(self, shape, maximum, scale_label):
        self.figure = Figure(figsize=(12, 5), dpi=80, facecolor="white")
        self.canvas = FigureCanvasAgg(self.figure)
        left = self.figure.add_axes((0.02, 0.18, 0.43, 0.64))
        self.source_axis = left
        right = self.figure.add_axes((0.48, 0.18, 0.43, 0.64))
        colour_axis = self.figure.add_axes((0.93, 0.20, 0.015, 0.60))
        self.source = left.imshow(np.zeros(shape), cmap="gray", vmin=0, vmax=255)
        cmap = colormaps["turbo_r"].with_extremes(bad="black")
        self.heatmap = right.imshow(np.full(shape, np.nan), cmap=cmap, vmin=0, vmax=maximum)
        self.figure.colorbar(self.heatmap, cax=colour_axis).set_label("Depth (cm)")
        left.set_title("Rectified first frame")
        right.set_title("Depth; black = invalid / unavailable")
        for axis in (left, right):
            axis.set_axis_off()
        self.title = self.figure.suptitle("")
        self.figure.text(0.5, 0.06, scale_label, ha="center", fontsize=10)
        self.figure.text(0.5, 0.015, "Only poses passing Step 5 quality checks are used.",
                         ha="center", fontsize=9)

    def draw(self, image, depth, title, rectified=True):
        self.source.set_data(image)
        self.source_axis.set_title("Rectified first frame" if rectified else
                                   "First frame (rectification unavailable)")
        self.heatmap.set_data(np.ma.masked_invalid(depth))
        self.title.set_text(title)
        self.canvas.draw()
        return cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)


def create_depth_video(video_path, K, D, calibration_size, pose_rows, pose_dir,
                       output_dir, baseline_cm=None, max_depth_cm=500.0):
    """Render every consecutive pair; flagged poses produce labelled blank maps.

    baseline_cm is the constant per-pair distance, not total camera travel.
    Without it, a 1 cm assumption is labelled, not claimed as a measurement.
    """
    assumed = baseline_cm is None
    baseline = 1.0 if assumed else float(baseline_cm)
    maximum = float(max_depth_cm)
    for name, value in (("baseline_cm", baseline), ("max_depth_cm", maximum)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    video_path, pose_dir, output_dir = map(Path, (video_path, pose_dir, output_dir))
    destination = output_dir / f"{video_path.stem.replace('_', '-')}.mp4"
    detail_dir = output_dir / f"{video_path.stem}_depth"
    detail_dir.mkdir(parents=True, exist_ok=True)
    label = ("ASSUMED SCALE: 1 cm per frame pair; not measured physical depth"
             if assumed else f"Configured baseline: {baseline:g} cm per frame pair")
    print(f"\nSTEP 6: {video_path.name}: {label}")
    capture = cv2.VideoCapture(str(video_path))
    writer, best = None, None
    rows, best_count = [], 0
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid FPS in {video_path}")
        ok, raw = capture.read()
        if not ok:
            raise ValueError(f"No frames in {video_path}")
        previous = _prepare_frame(raw, calibration_size, "ccw")
        plot = _DepthPlot(previous.shape, maximum, label)
        for index, pose in enumerate(pose_rows):
            ok, raw = capture.read()
            if not ok:
                raise RuntimeError("Video ended before the last pose pair")
            current = _prepare_frame(raw, calibration_size, "ccw")
            tag = f"frames_{index+1:03d}_{index+2:03d}"
            if pose["pair"] != tag:
                raise ValueError("Depth requires consecutive pose rows in source order")
            image = previous
            depth = np.full(previous.shape, np.nan, np.float32)
            status = "POSE_FLAGGED"
            if pose["status"] == "ACCEPTED":
                try:
                    R = np.loadtxt(pose_dir / f"{tag}_R.csv", delimiter=",")
                    t = np.loadtxt(pose_dir / f"{tag}_t.csv", delimiter=",")
                    image, disparity, depth, Q = calculate_depth(
                        previous, current, K, D, R, t, baseline, maximum,
                    )
                    count = int(np.isfinite(depth).sum())
                    status = "OK" if count else "NO_VALID_DISPARITY"
                    if count > best_count:
                        best_count = count
                        best = (tag, image.copy(), disparity, depth.copy(), Q)
                except (cv2.error, ValueError) as error:
                    status = f"RECTIFICATION_FAILED: {error}"
            count = int(np.isfinite(depth).sum())
            rows.append({"pair": tag, "status": status, "valid_pixels": count,
                         "valid_fraction": count / depth.size,
                         "median_depth_cm": float(np.nanmedian(depth)) if count else ""})
            frame = plot.draw(image, depth, f"{video_path.stem} | {tag} | {status.split(':')[0]}",
                              rectified=status in ("OK", "NO_VALID_DISPARITY"))
            if writer is None:
                writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise OSError(f"Cannot create {destination}")
            writer.write(frame)
            previous = current
        if capture.read()[0]:
            raise RuntimeError("Pose rows do not cover all consecutive frame pairs")
        if not rows:
            raise ValueError("No consecutive pairs to render")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    with (detail_dir / "depth_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    metadata = {"video": video_path.name, "output": destination.name,
                "baseline_cm": baseline, "scale_assumed": assumed, "scale_note": label,
                "max_depth_cm": maximum, "frames": len(rows),
                "frames_with_depth": sum(row["valid_pixels"] > 0 for row in rows),
                "method": "StereoSGBM; consistency <= 1.5 px; abs disparity >= 0.5 px",
                "best_pair": best[0] if best else None}
    if best is not None:
        tag, image, disparity, depth, Q = best
        np.savez_compressed(detail_dir / "best_depth.npz", depth_cm=depth,
                            disparity_px=disparity, Q=Q, pair=tag)
        if not cv2.imwrite(str(detail_dir / "best_depth.png"),
                           plot.draw(image, depth, f"{video_path.stem} | {tag}")):
            raise OSError("Cannot save depth preview")
    else:
        for name in ("best_depth.npz", "best_depth.png"):
            (detail_dir / name).unlink(missing_ok=True)
    (detail_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination.name}: {metadata['frames_with_depth']}/{len(rows)} pairs with valid depth")
    return metadata
