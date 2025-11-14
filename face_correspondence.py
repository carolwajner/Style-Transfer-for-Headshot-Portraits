import dlib
import numpy
import cv2


def _detect_landmarks(input: cv2.typing.MatLike, example: cv2.typing.MatLike) -> tuple:
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


def _affine_transform(
    from_points: list[tuple], to_points: list[tuple]
) -> cv2.typing.MatLike:
    # indexes of eyes (36 and 45) and nose tip (33)
    KEY_INDEXES = [36, 45, 33]

    from_key_points = [from_points[i] for i in KEY_INDEXES]
    to_key_points = [to_points[i] for i in KEY_INDEXES]

    from_points_np = numpy.array(from_key_points, dtype=numpy.float32)
    to_points_np = numpy.array(to_key_points, dtype=numpy.float32)

    matrix = cv2.getAffineTransform(from_points_np, to_points_np)
    return matrix


def morph_faces(input, example) -> cv2.typing.MatLike:
    points1, points2 = _detect_landmarks(input, example)

    matrix = _affine_transform(points2, points1)

    morphed_image = cv2.warpAffine(example, matrix, (input.shape[1], input.shape[0]))

    return morphed_image
