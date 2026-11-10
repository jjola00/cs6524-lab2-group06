
import glob
import numpy as np

from os import path as opath
from mypylibs import calibration

ROOT_IN = 'in'
ROOT_OUT = 'out'


def main(camera_id: str):

    video = opath.join(ROOT_IN, f'{camera_id}_calibration.mp4')
    image_points, image_size = calibration.find_corners(video)
    K, D, error = calibration.calibrate(image_points, image_size)
    calibration.save(K, D, camera_id, ROOT_OUT)

    np.set_printoptions(precision=4, suppress=True)
    print(f'K =\n{K}\nD (k1, k2, p1, p2, k3) = {D}\nmean reprojection error = {error:.4f} px')


if __name__ == '__main__':

    for video in sorted(glob.glob(opath.join(ROOT_IN, '*_calibration.mp4'))):
        camera_id = opath.basename(video)[:2]
        print(f'Processing camera {camera_id}')
        main(camera_id)
