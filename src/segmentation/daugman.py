import cv2
import numpy as np
import itertools
import math
from typing import Optional, Tuple, List
from .seg_utils import (
    get_square_bounding_box,
    get_refined_points,
    clean_gray_image,
    IrisData,
    DaugmanData
)
from config import Segmentation
from .fill_iris import flood_fill



def daugman(
    gray_img: np.ndarray,
    center: Tuple[int, int],
    start_r: int,
    end_r: int,
    step: int = 1,
) -> Tuple[float, int]:
    intensities = []
    mask = np.zeros_like(gray_img)

    # for every radius in range
    radii = list(range(start_r, end_r, step))  # type: List[int]
    for r in radii:
        # draw circle on mask
        cv2.circle(mask, center, r, 255, 1)
        # get pixel from original image, it is faster than np or cv2
        diff = gray_img & mask
        # normalize, np.add.reduce faster than .sum()
        #            diff[diff > 0] faster than .flatten()
        intensities.append(np.add.reduce(diff[diff > 0]) / (2 * math.pi * r))
        # refresh mask
        mask.fill(0)

    # calculate delta of radius insensitiveness
    #     mypy does not tolerate var type reload
    intensities_np = np.array(intensities, dtype=np.float32)
    del intensities

    # circles intensity differences, x5 faster than np.diff()
    intensities_np = intensities_np[:-1] - intensities_np[1:]
    intensities_np = abs(cv2.GaussianBlur(intensities_np, (1, 5), 0))
    # get maximum value
    idx = np.argmax(intensities_np)  # type: int

    return intensities_np[idx], radii[idx]


class Daugman:
    def __init__(self, config: Segmentation) -> None:
        self.config = config

    def segment(self, image: np.ndarray, eye_bb:np.ndarray) -> Optional[IrisData]:
        daugman_data = self.find_iris(image, eye_bb)
        if daugman_data is not None:
            x, y, w, h = eye_bb
            rel_iris_center = daugman_data.x_center - x, daugman_data.y_center - y
            eye = image[y : y + h, x : x + w]
            eye_mask = flood_fill(
                eye, rel_iris_center, daugman_data.radius, self.config.flood_fill
            )

            return IrisData(
                daugman_data=daugman_data,
                patch_im=eye,
                patch_bounding_box=(x,y,w,h),
                mask=eye_mask)


    def find_iris(
        self, image: np.ndarray, eye_bb: List
    ) -> Optional[DaugmanData]:

        daugman_config = self.config.daugman
        x, y, w, h = eye_bb
        daugman_start = int(h * daugman_config.min_radius_ratio)
        daugman_end = max(
            int(h * daugman_config.max_radius_ratio), daugman_config.min_radius_pixels
        )
        daugman_step = daugman_config.radius_step
        if daugman_config.force_square:
            x, y, w, h = get_square_bounding_box(eye_bb)
        eye = image[y : y + h, x : x + w]
        gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

        gray = clean_gray_image(gray, self.config.image_enhancement)
        h, w = gray.shape
        if daugman_config.reduce_points:
            points = get_refined_points(eye, self.config.image_enhancement)
            all_points = points[
                (points[:, 0] > h // 3)
                & (points[:, 0] < 2 * h // 3)
                & (points[:, 1] > w // 3)
                & (points[:, 1] < 2 * w // 3)
            ]
            all_points = all_points[:: daugman_config.points_step, :]

            if len(all_points) == 0:
                return None
        else:
            xrange = range(int(w / 3), w - int(w / 3), daugman_config.points_step)

            all_points = itertools.product(xrange, xrange)

        intensity_values = []
        coords = []  
        for point in all_points:
            val, r = daugman(gray, point, daugman_start, daugman_end, daugman_step)
            intensity_values.append(val)
            coords.append((point, r))


        best_idx = intensity_values.index(max(intensity_values))

        xcenter, ycenter = coords[best_idx][0][1] + x, coords[best_idx][0][0] + y
        radius = coords[best_idx][1]
        return DaugmanData(radius, xcenter, ycenter, gray)




