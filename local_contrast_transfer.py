import numpy
import cv2
from typing import List, Tuple


def create_laplacian_stacks(
    image: cv2.typing.MatLike, n: int
) -> Tuple[List[cv2.typing.MatLike], cv2.typing.MatLike]:
    laplacian_stack = []

    image = image.astype(numpy.float32)
    sigma = 1.5

    for i in range(n):
        if i == 0:
            L0 = image - cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
            laplacian_stack.append(L0)
        else:
            Li = cv2.GaussianBlur(
                image, (0, 0), sigmaX=sigma**i, sigmaY=sigma**i
            ) - cv2.GaussianBlur(
                image, (0, 0), sigmaX=sigma ** (i + 1), sigmaY=sigma ** (i + 1)
            )
            laplacian_stack.append(Li)

    residual = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma**n, sigmaY=sigma**n)

    return laplacian_stack, residual
