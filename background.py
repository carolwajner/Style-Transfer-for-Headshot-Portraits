import cv2
import numpy as np

from rembg import remove, new_session

session = new_session("u2net")

def get_segmentation_mask(
    img: cv2.typing.MatLike, 
) -> cv2.typing.MatLike:
    
    result = remove(img, session=session)
    
    mask = result[:, :, 3]
    
    _, mask_binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    
    return mask_binary


def inpainting_background_extraction(
    img: cv2.typing.MatLike, 
    subject_mask: cv2.typing.MatLike
) -> cv2.typing.MatLike:

    dilated_mask = cv2.dilate(subject_mask, None, iterations=3)

    clean_bg = cv2.inpaint(img, dilated_mask, 5, cv2.INPAINT_TELEA)
    return clean_bg


def replace_background(
    foreground_img: cv2.typing.MatLike,
    background_img: cv2.typing.MatLike,
    mask: cv2.typing.MatLike
) -> cv2.typing.MatLike:

    h, w = foreground_img.shape[:2]
    
    bg_resized = cv2.resize(background_img, (w, h))
    
    mask_float = mask.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (3, 3), 0)
    mask_3ch = cv2.merge([mask_float, mask_float, mask_float])
    
    fg_float = foreground_img.astype(np.float32) / 255.0
    bg_float = bg_resized.astype(np.float32) / 255.0
    
    final_blended = (fg_float * mask_3ch) + (bg_float * (1.0 - mask_3ch))
    
    return (final_blended * 255).clip(0, 255).astype(np.uint8)