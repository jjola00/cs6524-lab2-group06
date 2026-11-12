
import cv2
import numpy as np

HEADER = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header"""


def save_ply(path, depth_cm, colour_bgr, K, step=2, max_jump=0.05):
    """Bonus: write one depth frame as a coloured point cloud (.ply, cm).

    depth_cm   HxW depth from Step 6 (0 / NaN / inf = no depth)
    colour_bgr HxWx3 frame the depth belongs to (the rectified left frame)
    K          3x3 intrinsics of that frame (after rectification: P1[:, :3])
    step       keep every `step`-th pixel, to keep the file light
    max_jump   drop pixels whose 3x3 neighbourhood varies more than this
               fraction of their depth (removes the streaks at object edges)
    """
    if colour_bgr.ndim == 2:  # greyscale frame -> grey points
        colour_bgr = cv2.cvtColor(colour_bgr, cv2.COLOR_GRAY2BGR)
    depth = np.nan_to_num(depth_cm.astype(np.float32), nan=0, posinf=0, neginf=0)
    valid = depth > 0

    # Drop the extreme 2% at each end (bad disparities)
    low, high = np.percentile(depth[valid], [2, 98])
    valid &= (depth >= low) & (depth <= high)

    # Drop "flying pixels" between foreground and background
    kernel = np.ones((3, 3), np.uint8)
    spread = cv2.dilate(depth, kernel) - cv2.erode(depth, kernel)
    valid &= spread <= max_jump * depth

    # Keep a regular subsample of the remaining pixels
    grid = np.zeros_like(valid)
    grid[::step, ::step] = True
    v, u = np.nonzero(valid & grid)

    # Back-project pixel (u, v) at depth Z:  X = (u - cx) Z / fx,  Y = (v - cy) Z / fy
    Z = depth[v, u]
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    rgb = colour_bgr[v, u][:, ::-1]

    # Flip Y and Z so the scene is upright and faces the viewer in CloudCompare
    points = np.column_stack([X, -Y, -Z, rgb])
    np.savetxt(path, points, fmt='%.2f %.2f %.2f %d %d %d', header=HEADER.format(len(points)), comments='')
    return len(points)
