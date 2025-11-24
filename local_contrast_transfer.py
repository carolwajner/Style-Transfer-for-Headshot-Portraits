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
                image, (0, 0), sigmaX=sigma ** i, sigmaY=sigma ** i
            ) - cv2.GaussianBlur(
                image, (0, 0), sigmaX=sigma ** (i + 1), sigmaY=sigma ** (i + 1)
            )
            laplacian_stack.append(Li)

    residual = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma ** n, sigmaY=sigma ** n)

    return laplacian_stack, residual


def compute_local_energy(
        laplacian_stack: List[cv2.typing.MatLike],
) -> List[cv2.typing.MatLike]:

    energy_stack = []

    for level, layer in enumerate(laplacian_stack):
        squared_layer = numpy.square(layer)

        sigma_energy = 2 ** (level + 1)

        energy = cv2.GaussianBlur(
            squared_layer, (0, 0), sigmaX=sigma_energy, sigmaY=sigma_energy
        )

        energy = numpy.maximum(energy, 1e-6)

        energy_stack.append(energy)

    return energy_stack


def _compute_robust_gain(
        energy_in: cv2.typing.MatLike,
        energy_ex: cv2.typing.MatLike,
        level: int
) -> cv2.typing.MatLike:

    # constantes ideais do paper
    EPSILON = 0.01 ** 2
    THETA_H = 2.8
    THETA_L = 0.9
    BETA = 3.0

    gain = numpy.sqrt(energy_ex / (energy_in + EPSILON))

    robust_gain = numpy.clip(gain, THETA_L, THETA_H)

    sigma_gain = BETA * (2 ** level)

    robust_gain = cv2.GaussianBlur(robust_gain, (0, 0), sigmaX=sigma_gain, sigmaY=sigma_gain)

    return robust_gain


def transfer_style(
        input_stack: list[cv2.typing.MatLike],
        energy_in: list[cv2.typing.MatLike],
        energy_ex: list[cv2.typing.MatLike]
) -> Tuple[List[cv2.typing.MatLike], List[cv2.typing.MatLike]]:

    output_stack = []

    for level, layer_in in enumerate(input_stack):
        gain_map = _compute_robust_gain(energy_in[level], energy_ex[level], level)

        new_layer = layer_in * gain_map
        output_stack.append(new_layer)

    return output_stack


def reconstruct_image(
        laplacian_stack: list[cv2.typing.MatLike],
        residual: cv2.typing.MatLike
) -> cv2.typing.MatLike:

    final_image = residual.copy()

    for layer in laplacian_stack:
        final_image += layer

    final_image = numpy.clip(final_image, 0.0, 1.0)

    return final_image
