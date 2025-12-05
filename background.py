import cv2
import numpy as np
from typing import List, Tuple

def __get_extended_hull_mask(
    shape: Tuple[int, int], 
    landmarks: List[Tuple[int, int]]
) -> Tuple[cv2.typing.MatLike, np.ndarray]:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    points = np.array(landmarks, dtype=np.int32)
    
    eyebrow_left_outer = points[17]
    eyebrow_right_outer = points[26]
    nose_top = points[27]
    
    face_height = np.linalg.norm(points[8] - points[27]) 
    forehead_height = face_height * 0.8 
    
    p1 = (int(eyebrow_left_outer[0]), int(eyebrow_left_outer[1] - forehead_height))
    p2 = (int(eyebrow_right_outer[0]), int(eyebrow_right_outer[1] - forehead_height))
    p3 = (int(nose_top[0]), int(nose_top[1] - forehead_height * 1.1))
    
    extended_points = np.vstack([points, np.array([p1, p2, p3], dtype=np.int32)])
    
    hull = cv2.convexHull(extended_points)
    
    cv2.fillConvexPoly(mask, hull, cv2.GC_FGD) 
    
    return mask, hull

def __get_segmentation_mask(
    img: cv2.typing.MatLike, 
    landmarks: List[Tuple[int, int]]
) -> cv2.typing.MatLike:
    height, width = img.shape[:2]
    mask = np.zeros(img.shape[:2], np.uint8)

    points = np.array(landmarks, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(points)
    
    pad_w = int(w * 0.3)     
    pad_h_top = int(h * 0.8)  
    pad_h_bot = int(h * 0.5) 
    
    rect_x = x - pad_w
    rect_y = y - pad_h_top
    rect_w = w + 2 * pad_w
    rect_h = h + pad_h_top + pad_h_bot
    
    rect_x = max(0, rect_x)
    rect_y = max(0, rect_y)
    rect_w = min(width - rect_x, rect_w)
    rect_h = min(height - rect_y, rect_h)

    if rect_w >= width:
        rect_x += 1
        rect_w -= 2
    if rect_h >= height:
        rect_y += 1
        rect_h -= 2

    rect_w = max(1, rect_w)
    rect_h = max(1, rect_h)
    
    rect = (int(rect_x), int(rect_y), int(rect_w), int(rect_h))

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        _, hull = __get_extended_hull_mask(img.shape, landmarks)
        cv2.fillConvexPoly(mask, hull, cv2.GC_FGD)
        
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        
    except cv2.error as e:
        print(f"Warning: GrabCut failed (likely rect size issue). Using Convex Hull fallback. Error: {e}")
        mask.fill(0)
        _, hull = __get_extended_hull_mask(img.shape, landmarks)
        cv2.fillConvexPoly(mask, hull, 255) 
        return mask

    mask_binary = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    
    return mask_binary

def __inpainting_background_extraction(
    img: cv2.typing.MatLike, 
    subject_mask: cv2.typing.MatLike
) -> cv2.typing.MatLike:
    dilated_mask = cv2.dilate(subject_mask, None, iterations=10)
    clean_bg = cv2.inpaint(img, dilated_mask, 5, cv2.INPAINT_TELEA)
    return clean_bg

def replace_background(
    input_img: cv2.typing.MatLike, 
    example_img: cv2.typing.MatLike, 
    input_landmarks: List[Tuple[int, int]], 
    example_landmarks: List[Tuple[int, int]]
) -> cv2.typing.MatLike:
    
    input_fg_mask = __get_segmentation_mask(input_img, input_landmarks)
    
    example_fg_mask = __get_segmentation_mask(example_img, example_landmarks)
    
    clean_example_bg = __inpainting_background_extraction(example_img, example_fg_mask)
    
    h, w = input_img.shape[:2]
    clean_example_bg_resized = cv2.resize(clean_example_bg, (w, h))
    
    mask_float = input_fg_mask.astype(float) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (9, 9), 0) # Soften edges
    alpha = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)
    
    foreground = input_img.astype(float) * alpha
    background = clean_example_bg_resized.astype(float) * (1.0 - alpha)
    
    final_image = (foreground + background).astype(np.uint8)
    
    return final_image