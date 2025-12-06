import dlib
import numpy
import cv2
from typing import Tuple, List

import morphing


def __add_synthetic_forehead(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    
    points_np = numpy.array(points, dtype=numpy.int32)
    
    nose_bridge = points_np[27] 
    chin = points_np[8]
    
    face_height = numpy.linalg.norm(chin - nose_bridge)
    new_points = []
    steps = 12 
    
    for i in range(steps + 1):
        t = i / steps
        current_angle = numpy.deg2rad(180 - (t * 180))
        
        radius = face_height * 0.5
        
        dx = radius * numpy.cos(current_angle)
        dy = -radius * numpy.sin(current_angle)
        
        x = int(nose_bridge[0] + dx)
        y = int(nose_bridge[1] + dy)
        new_points.append((x, y))
        
    return points + new_points


def __auto_detect_hairline(
    img: cv2.typing.MatLike, 
    points: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    
    points_np = numpy.array(points, dtype=numpy.int32)
    height, width = img.shape[:2]
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    nose_bridge = points[27]
    
    forehead_indices = range(68, len(points))

    for i in forehead_indices:
        start_x, start_y = points[i]
        
        dx = start_x - nose_bridge[0]
        dy = start_y - nose_bridge[1]
        
        length = numpy.sqrt(dx*dx + dy*dy)
        if length == 0: continue
        dir_x, dir_y = dx/length, dy/length
        
        ref_sample = hsv_img[max(0, start_y-2):min(height, start_y+3), 
                             max(0, start_x-2):min(width, start_x+3)]
        if ref_sample.size == 0: continue
        ref_color = numpy.mean(ref_sample, axis=(0,1))
        
        current_x, current_y = float(start_x), float(start_y)
        
        max_dist = length * 2.0 
        traveled = 0
        hit_hair = False
        
        while traveled < max_dist:
            current_x += dir_x
            current_y += dir_y
            traveled += 1
            
            ix, iy = int(current_x), int(current_y)
            if ix < 0 or ix >= width or iy < 0 or iy >= height:
                break
            
            pixel_color = hsv_img[iy, ix]
            
            diff_h = abs(float(pixel_color[0]) - float(ref_color[0]))
            if diff_h > 90: diff_h = 180 - diff_h
            diff_s = abs(float(pixel_color[1]) - float(ref_color[1]))
            diff_v = abs(float(pixel_color[2]) - float(ref_color[2]))
            
            total_diff = (diff_h * 2) + (diff_s * 0.5) + (diff_v * 1.0)
            
            if total_diff > 40: 
                hit_hair = True
                break
        
        if hit_hair:
            final_x = current_x - (dir_x * 5)
            final_y = current_y - (dir_y * 5)
            points_np[i] = (int(final_x), int(final_y))
        else:
            points_np[i] = (int(current_x), int(current_y))

    return [(int(p[0]), int(p[1])) for p in points_np]


def detect_landmarks(
    input: cv2.typing.MatLike, example: cv2.typing.MatLike
) -> Tuple[List, List]:
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    input_face = detector(input, 1)
    example_face = detector(example, 1)

    if len(input_face) == 0 or len(example_face) == 0:
        raise ValueError("No faces detected in one of the images.")

    landmarks1 = predictor(input, input_face[0])
    landmarks2 = predictor(example, example_face[0])

    points1 = [(landmarks1.part(i).x, landmarks1.part(i).y) for i in range(68)]
    points2 = [(landmarks2.part(i).x, landmarks2.part(i).y) for i in range(68)]

    points1 = __add_synthetic_forehead(points1)
    points2 = __add_synthetic_forehead(points2)

    points1 = __auto_detect_hairline(input, points1)
    points2 = __auto_detect_hairline(example, points2)

    return points1, points2


def __affine_transform(
    from_points: List[Tuple], to_points: List[Tuple]
) -> cv2.typing.MatLike:
    # indexes of eyes (36 and 45) mouth (51)
    KEY_INDEXES = [36, 45, 51]

    from_key_points = [from_points[i] for i in KEY_INDEXES]
    to_key_points = [to_points[i] for i in KEY_INDEXES]

    from_points_np = numpy.array(from_key_points, dtype=numpy.float32)
    to_points_np = numpy.array(to_key_points, dtype=numpy.float32)

    matrix = cv2.getAffineTransform(from_points_np, to_points_np)
    return matrix


def __get_structure(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(
        numpy.uint8
    )
    return gradient


def __apply_adjustment(
    target_img: cv2.typing.MatLike, source_img: cv2.typing.MatLike
) -> cv2.typing.MatLike:
    target_struct = __get_structure(target_img)
    source_struct = __get_structure(source_img)

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
        flags=0,
    )

    flow = cv2.GaussianBlur(flow, (0, 0), 5.0)

    h, w = flow.shape[:2]
    grid_y, grid_x = numpy.mgrid[0:h, 0:w].astype(numpy.float32)

    map_x = grid_x + (flow[..., 0] * 0.6)
    map_y = grid_y + (flow[..., 1] * 0.6)

    final_img = cv2.remap(
        source_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return final_img


def dense_morph(
    input: cv2.typing.MatLike, example: cv2.typing.MatLike
) -> cv2.typing.MatLike:
    # Detect 68 landmarks
    points_input, points_example = detect_landmarks(input, example)

    # Align eyes and mouth using affine transform
    affine_matrix = __affine_transform(points_example, points_input)

    aligned_example = cv2.warpAffine(
        example,
        affine_matrix,
        (input.shape[1], input.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Beier-Neely morphing
    points_aligned_example, _ = detect_landmarks(aligned_example, input)
    morphed_output = morphing.beier_neely(
        aligned_example, points_aligned_example, points_input
    )

    # SIFT flow adjustment
    final_output = __apply_adjustment(input, morphed_output)

    return final_output