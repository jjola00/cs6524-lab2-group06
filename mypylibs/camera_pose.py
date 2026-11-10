
"""Step 5: relative camera pose from SIFT matches in consecutive video frames.

Scene videos may be portrait while calibration is landscape. Portrait frames are
rotated to landscape and resized to the calibration image size before analysis.
"""

from pathlib import Path
import csv
import cv2
import numpy as np


def _prepare_frame(frame, calibration_size, portrait_rotation):
    width, height = calibration_size
    h, w = frame.shape[:2]

    if h > w and width > height:
        code = (
            cv2.ROTATE_90_COUNTERCLOCKWISE
            if portrait_rotation == "ccw"
            else cv2.ROTATE_90_CLOCKWISE
        )
        frame = cv2.rotate(frame, code)

    elif (h > w) != (height > width):
        raise ValueError(
            "Video and calibration orientations differ unexpectedly"
        )

    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def estimate_camera_pose(
    video_path,
    K,
    D,
    output_dir,
    calibration_size=None,
    frame_step=1,
    portrait_rotation="ccw",
    max_rotation_degrees=10.0
):
    """
    Save F, E, R and unit-length t for frame pairs,
    plus a summary CSV.

    frame_step=1 means consecutive frames.
    t has unknown metric scale.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1)

    if portrait_rotation not in ("ccw", "cw"):
        raise ValueError(
            "portrait_rotation must be 'ccw' or 'cw'"
        )

    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Cannot open video: {video_path}"
        )

    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if calibration_size is None:
        calibration_size = (
            max(raw_width, raw_height),
            min(raw_width, raw_height)
        )

    calibration_size = tuple(map(int, calibration_size))

    if (
        raw_width != calibration_size[0]
        or raw_height != calibration_size[1]
    ):
        print(
            f"  Aligning {raw_width}x{raw_height} frames to "
            f"{calibration_size[0]}x{calibration_size[1]} "
            f"calibration coordinates"
        )

    sift = cv2.SIFT_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    frames = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frames.append(
            _prepare_frame(
                frame,
                calibration_size,
                portrait_rotation
            )
        )

    cap.release()

    if len(frames) <= frame_step:
        raise ValueError(
            f"Not enough frames in {video_path}"
        )

    features = [
        sift.detectAndCompute(frame, None)
        for frame in frames
    ]

    rows = []

    for i in range(len(frames) - frame_step):

        j = i + frame_step

        tag = f"frames_{i+1:03d}_{j+1:03d}"

        kp1, desc1 = features[i]
        kp2, desc2 = features[j]

        row = dict(
            pair=tag,
            matches=0,
            essential_inliers=0,
            pose_inliers=0,
            pose_ratio=0.0,
            rotation_deg="",
            tx="",
            ty="",
            tz="",
            status="FLAGGED"
        )

        if (
            desc1 is None
            or desc2 is None
            or len(desc1) < 2
            or len(desc2) < 2
        ):
            rows.append(row)
            continue

        pairs = matcher.knnMatch(
            desc1,
            desc2,
            k=2
        )

        good = [
            m for m, n in pairs
            if m.distance < 0.75 * n.distance
        ]

        row["matches"] = len(good)

        if len(good) < 12:
            rows.append(row)
            continue

        p1 = np.float64([
            kp1[m.queryIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        p2 = np.float64([
            kp2[m.trainIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        # Correct lens distortion.
        # P=K keeps the coordinates in pixels.

        p1 = cv2.undistortPoints(
            p1, K, D, P=K
        ).reshape(-1, 2)

        p2 = cv2.undistortPoints(
            p2, K, D, P=K
        ).reshape(-1, 2)

        # Estimate the Essential Matrix directly.

        E, mask = cv2.findEssentialMat(
            p1,
            p2,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None or mask is None:
            rows.append(row)
            continue

        # OpenCV can return multiple candidate matrices.

        candidates = np.asarray(E).reshape(-1, 3, 3)

        best = None

        for candidate in candidates:

            try:
                count, R, t, pose_mask = cv2.recoverPose(
                    candidate,
                    p1,
                    p2,
                    K,
                    mask=mask.copy()
                )

            except cv2.error:
                continue

            if best is None or count > best[0]:
                best = (
                    count,
                    candidate,
                    R,
                    t
                )

        if best is None:
            rows.append(row)
            continue

        count, E, R, t = best

        n_inliers = int(np.count_nonzero(mask))

        ratio = count / max(n_inliers, 1)

        angle = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        (np.trace(R) - 1) / 2,
                        -1,
                        1
                    )
                )
            )
        )

        # Calculate F from E and K.

        invK = np.linalg.inv(K)

        F = invK.T @ E @ invK

        # Save the matrices for each frame pair.

        for name, matrix in (
            ("F", F),
            ("E", E),
            ("R", R),
            ("t", t)
        ):
            np.savetxt(
                output_dir / f"{tag}_{name}.csv",
                matrix,
                delimiter=","
            )

        # Flag potentially unreliable poses.

        accepted = (
            count >= 20
            and ratio >= 0.5
            and angle <= max_rotation_degrees
        )

        row.update(
            essential_inliers=n_inliers,
            pose_inliers=int(count),
            pose_ratio=round(ratio, 4),
            rotation_deg=round(angle, 4),
            tx=round(float(t[0, 0]), 5),
            ty=round(float(t[1, 0]), 5),
            tz=round(float(t[2, 0]), 5),
            status="ACCEPTED" if accepted else "FLAGGED"
        )

        rows.append(row)

        if i < 3 or (i + 1) % 10 == 0:
            print(
                f"  {tag}: matches={len(good)}, "
                f"E_inliers={n_inliers}, "
                f"pose_inliers={count}, "
                f"rotation={angle:.2f}°, "
                f"{row['status']}"
            )

    # Save the summary CSV.

    summary = output_dir / "pose_summary.csv"

    with summary.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)

    accepted_count = sum(
        row["status"] == "ACCEPTED"
        for row in rows
    )

    print(f"\nVideo: {video_path.name}")

    print(
        f"Frame pairs: {len(rows)} | "
        f"Accepted: {accepted_count} | "
        f"Flagged: {len(rows) - accepted_count}"
    )

    print(f"Results saved to: {summary}")

    return rows
