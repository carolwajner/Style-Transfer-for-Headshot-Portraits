""" Adaptação do código: https://github.com/antonmdv/Morphing/blob/master/morphing.py"""

import numpy as np
import cv2


FACE_SEGMENTS: list[tuple[int, int]] = [
    # Mandíbula
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    # Sobrancelhas: 17 a 21 (esquerda) e 22 a 26 (direita)
    (17, 18), (18, 19), (19, 20), (20, 21),
    (22, 23), (23, 24), (24, 25), (25, 26),
    # Nariz: 27 a 30 (ponte) e 31 a 35 (narinas)
    (27, 28), (28, 29), (29, 30),
    (31, 32), (32, 33), (33, 34), (34, 35),
    # Olhos: 36 a 41 (esquerdo) e 42 a 47 (direito)
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    # Boca: 48 a 59 (parte externa) e 60 a 67 (parte interna)
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59),
    (59, 48),
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)
]


def __perpendicular(v: np.ndarray):
    return np.array([-v[1], v[0]])


def __get_lines(landmarks: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    P = []
    Q = []
    for idx1, idx2 in FACE_SEGMENTS:
        P.append(landmarks[idx1])
        Q.append(landmarks[idx2])
    return np.array(P), np.array(Q)


def beier_neely(src_img: cv2.typing.MatLike, src_landmarks: list, dest_landmarks: list) -> cv2.typing.MatLike:
    height, width = src_img.shape[:2]
    dest_img = np.zeros_like(src_img)

    P_src, Q_src = __get_lines(src_landmarks)
    P_dest, Q_dest = __get_lines(dest_landmarks)

    QP_dest = Q_dest - P_dest
    len_QP_dest_sq = np.sum(QP_dest ** 2, axis=1)
    len_QP_dest = np.sqrt(len_QP_dest_sq)
    perp_QP_dest = np.array([__perpendicular(v) for v in QP_dest])

    QP_src = Q_src - P_src
    len_QP_src = np.sqrt(np.sum(QP_src ** 2, axis=1))
    perp_QP_src = np.array([__perpendicular(v) for v in QP_src])

    # Constantes de Beier e Neely [1992]
    a = 0.001
    b = 2.0
    p = 0.5

    for y in range(height):
        for x in range(width):
            X = np.array([x, y])

            DSUM = np.array([0.0, 0.0])
            weight_sum = 0.0

            X_minus_P = X - P_dest

            u = np.sum(X_minus_P * QP_dest, axis=1) / len_QP_dest_sq
            v = np.sum(X_minus_P * perp_QP_dest, axis=1) / len_QP_dest

            X_prime = P_src + (u[:, np.newaxis] * QP_src) + (v[:, np.newaxis] * perp_QP_src) / len_QP_src[:, np.newaxis]

            D = X_prime - X

            dist = np.zeros_like(u)

            mask_less = u < 0
            if np.any(mask_less):
                dist[mask_less] = np.sqrt(np.sum((X - P_dest[mask_less]) ** 2, axis=1))

            mask_more = u > 1
            if np.any(mask_more):
                dist[mask_more] = np.sqrt(np.sum((X - Q_dest[mask_more]) ** 2, axis=1))

            mask_mid = (~mask_less) & (~mask_more)
            dist[mask_mid] = np.abs(v[mask_mid])

            weight = np.power((np.power(len_QP_dest, p) / (a + dist)), b)

            DSUM = np.sum(D * weight[:, np.newaxis], axis=0)
            weight_sum = np.sum(weight)

            if weight_sum > 0:
                X_final = X + DSUM / weight_sum
            else:
                X_final = X

            srcX, srcY = int(X_final[0]), int(X_final[1])

            if 0 <= srcX < width and 0 <= srcY < height:
                dest_img[y, x] = src_img[srcY, srcX]
            else:
                dest_img[y, x] = [0, 0, 0]

    return dest_img