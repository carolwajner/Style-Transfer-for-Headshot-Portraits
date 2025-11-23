import dlib
import numpy
import cv2

import morphing


def __detect_landmarks(input: cv2.typing.MatLike, example: cv2.typing.MatLike) -> tuple:
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

    return points1, points2


def __affine_transform(
    from_points: list[tuple], to_points: list[tuple]
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


def dense_morph(input, example) -> cv2.typing.MatLike:
    # Detect 68 landmarks
    points_input, points_example = __detect_landmarks(input, example)

    # Align eyes and mouth using affine transform
    affine_matrix = __affine_transform(points_example, points_input)

    aligned_example = cv2.warpAffine(
        example,
        affine_matrix,
        (input.shape[1], input.shape[0]),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Beier-Neely morphing
    points_aligned_example, _ = __detect_landmarks(aligned_example, input)
    morphed_output = morphing.beier_neely(
        aligned_example, points_aligned_example, points_input
    )

    # SIFT flow adjustment
    final_output = __apply_adjustment(input, morphed_output)

    return final_output
