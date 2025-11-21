import cv2
import numpy as np


def _get_structure(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gradient


def apply_adjustment(target_img: cv2.typing.MatLike, source_img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    target_struct = _get_structure(target_img)
    source_struct = _get_structure(source_img)

    flow = cv2.calcOpticalFlowFarneback(
        target_struct,
        source_struct,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=40,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=0
    )

    flow = cv2.GaussianBlur(flow, (0, 0), 5.0)

    h, w = flow.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    map_x = grid_x + (flow[..., 0] * 0.6)
    map_y = grid_y + (flow[..., 1] * 0.6)

    final_img = cv2.remap(source_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return final_img