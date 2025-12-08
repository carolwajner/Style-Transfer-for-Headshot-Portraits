import numpy
import cv2
from typing import List, Tuple

def __create_laplacian_stacks(
    image: cv2.typing.MatLike, n: int
) -> Tuple[List[cv2.typing.MatLike], cv2.typing.MatLike]:
    laplacian_stack = []

    image = image.astype(numpy.float32)
    sigma = 2

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


def __compute_local_energy(
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


def __compute_robust_gain(
    energy_in: cv2.typing.MatLike, energy_ex: cv2.typing.MatLike, level: int
) -> cv2.typing.MatLike:

    # constantes ideais do paper
    EPSILON = 0.01**2
    THETA_H = 2.8
    THETA_L = 0.9
    BETA = 3.0

    gain = numpy.sqrt(energy_ex / (energy_in + EPSILON))

    robust_gain = numpy.clip(gain, THETA_L, THETA_H)

    sigma_gain = BETA * (2**level)

    robust_gain = cv2.GaussianBlur(
        robust_gain, (0, 0), sigmaX=sigma_gain, sigmaY=sigma_gain
    )

    return robust_gain


def __transfer_style(
    input_stack: list[cv2.typing.MatLike],
    energy_in: list[cv2.typing.MatLike],
    energy_ex: list[cv2.typing.MatLike],
) -> Tuple[List[cv2.typing.MatLike], List[cv2.typing.MatLike]]:

    output_stack = []

    for level, layer_in in enumerate(input_stack):
        gain_map = __compute_robust_gain(energy_in[level], energy_ex[level], level)

        new_layer = layer_in * gain_map
        output_stack.append(new_layer)

    return output_stack


def __reconstruct_image(
    laplacian_stack: list[cv2.typing.MatLike], residual: cv2.typing.MatLike
) -> cv2.typing.MatLike:

    final_image = residual.copy()

    for layer in laplacian_stack:
        final_image += layer

    final_image = numpy.clip(final_image, 0.0, 1.0)

    return final_image


def __apply_color_transfer(
    final_luminance: cv2.typing.MatLike,
    input_color_img: cv2.typing.MatLike
) -> cv2.typing.MatLike:

    h, w = final_luminance.shape[:2]
    input_resized = cv2.resize(input_color_img, (w, h))

    lab_example = cv2.cvtColor(input_resized, cv2.COLOR_BGR2LAB)

    _, a_ex, b_ex = cv2.split(lab_example)

    if final_luminance.dtype != numpy.uint8:
        l_new = (final_luminance * 255).clip(0, 255).astype(numpy.uint8)
    else:
        l_new = final_luminance

    result_lab = cv2.merge([l_new, a_ex, b_ex])

    result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    return result_rgb

def __extend_morphed_colors(
    morphed_img: cv2.typing.MatLike
) -> cv2.typing.MatLike:

    lab_morph = cv2.cvtColor(morphed_img, cv2.COLOR_BGR2LAB)
    a = lab_morph[:,:,1].astype(float)
    b = lab_morph[:,:,2].astype(float)

    saturation_dist = numpy.sqrt((a - 128.0)**2 + (b - 128.0)**2)
    
    gray_mask = (saturation_dist < 13).astype(numpy.uint8) * 255
    
    gray_mask = cv2.dilate(gray_mask, None, iterations=2)
    
    extended_colors = cv2.inpaint(morphed_img, gray_mask, 5, cv2.INPAINT_TELEA)
    
    return extended_colors


def apply_local_contrast_and_blend(
    input_img: cv2.typing.MatLike,
    morphed_img: cv2.typing.MatLike,
    input_img_color: cv2.typing.MatLike,
    segmentation_mask: cv2.typing.MatLike,
    clean_background: cv2.typing.MatLike, 
    stack_levels: int = 7,
    blur_radius: int = 21, # <--- 1. NEW PARAMETER (Must be an odd number)
) -> cv2.typing.MatLike:

    h, w = input_img.shape[:2]

    mask_float = segmentation_mask.astype(numpy.float32) / 255.0
    
    mask_float = cv2.GaussianBlur(mask_float, (blur_radius, blur_radius), 0)
    
    mask_3ch = cv2.merge([mask_float, mask_float, mask_float])

    bg_ex = cv2.resize(clean_background, (w, h)).astype(numpy.float32) / 255.0

    input_rgb_float = input_img_color.astype(numpy.float32) / 255.0
    
    input_composited = (input_rgb_float * mask_3ch) + (bg_ex * (1.0 - mask_3ch))

    # tons de cinza
    input_gray = cv2.cvtColor(input_composited.astype(numpy.float32), cv2.COLOR_BGR2GRAY)
    example_gray = (
        cv2.cvtColor(morphed_img, cv2.COLOR_BGR2GRAY).astype(numpy.float32) / 255.0
    )

    # pilhas laplacianas
    stacks_input, _ = __create_laplacian_stacks(input_gray, stack_levels)
    stacks_output, residual_output = __create_laplacian_stacks(example_gray, stack_levels)

    # energia local
    energy_input = __compute_local_energy(stacks_input)
    energy_output = __compute_local_energy(stacks_output)

    # mapas de calor
    new_stack = __transfer_style(stacks_input, energy_input, energy_output)

    final_result_gray = __reconstruct_image(new_stack, residual_output)

    # colorir partes não usadas no morph
    extended_color_source = __extend_morphed_colors(morphed_img)
    
    # reconstrução do res do zuck pra iluminação base
    final_result_color = __apply_color_transfer(
        final_result_gray, extended_color_source
    )

    # remoção do fundo
    final_result_float = final_result_color.astype(numpy.float32) / 255.0
    
    final_blended = (final_result_float * mask_3ch) + (bg_ex * (1.0 - mask_3ch))

    return (final_blended * 255).clip(0, 255).astype(numpy.uint8)