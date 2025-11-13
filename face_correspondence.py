import dlib


def get_face_correspondence(image1, image2):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    faces1 = detector(image1, 1)
    faces2 = detector(image2, 1)

    if len(faces1) == 0 or len(faces2) == 0:
        raise ValueError("No faces detected in one of the images.")

    landmarks1 = predictor(image1, faces1[0])
    landmarks2 = predictor(image2, faces2[0])

    points1 = [(landmarks1.part(i).x, landmarks1.part(i).y) for i in range(68)]
    points2 = [(landmarks2.part(i).x, landmarks2.part(i).y) for i in range(68)]

    return points1, points2
