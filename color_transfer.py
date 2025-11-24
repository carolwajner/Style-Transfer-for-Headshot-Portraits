import cv2
import numpy as np


def apply_color_transfer(
    final_luminance: cv2.typing.MatLike,
    input_color_img: cv2.typing.MatLike
) -> cv2.typing.MatLike:

    h, w = final_luminance.shape[:2]
    input_resized = cv2.resize(input_color_img, (w, h))

    lab_example = cv2.cvtColor(input_resized, cv2.COLOR_BGR2LAB)

    # l_in, a_in, b_in = cv2.split(lab_input)
    _, a_ex, b_ex = cv2.split(lab_example)

    if final_luminance.dtype != np.uint8:
        l_new = (final_luminance * 255).clip(0, 255).astype(np.uint8)
    else:
        l_new = final_luminance

    result_lab = cv2.merge([l_new, a_ex, b_ex])

    result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    return result_rgb
