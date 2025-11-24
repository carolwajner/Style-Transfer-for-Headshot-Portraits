import sys
import utils
import dense_correspondence
import local_contrast_transfer
import color_transfer
import numpy
import cv2


def main(arg1: str, arg2: str):
    try:
        input_img = utils.read_image(arg1)
        example_img = utils.read_image(arg2)

        # backup da imagem pq deu errado sem isso e com isso tbm da, mass fica melhor
        # o problema é q a camiseta do rapaz fica azul
        input_img_color = input_img.copy()

        # geometria e deformação
        output = dense_correspondence.dense_morph(input_img, example_img)

        # tons de cinza
        input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY).astype(numpy.float32) / 255.0
        example_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(numpy.float32) / 255.0

        # pilhas laplacianas
        stacks_input, residual_input = local_contrast_transfer.create_laplacian_stacks(input_gray, 7)
        stacks_output, residual_output = local_contrast_transfer.create_laplacian_stacks(example_gray, 7)

        # energia local
        energy_input = local_contrast_transfer.compute_local_energy(stacks_input)
        energy_output = local_contrast_transfer.compute_local_energy(stacks_output)

        # mapas de calor
        new_stack = local_contrast_transfer.transfer_style(stacks_input, energy_input, energy_output)

        # reconstrução do res do zuck pra iluminação base
        final_result_gray = local_contrast_transfer.reconstruct_image(new_stack, residual_output)
        gray_uint8 = (final_result_gray * 255).clip(0, 255).astype(numpy.uint8)
        utils.save_image(gray_uint8, "data/final_result_gray.jpg")

        # cor lab do zuck deformado (output)
        final_result_color = color_transfer.apply_color_transfer(final_result_gray, output)

        # tentativa FALHA de arrumar a cor da orelha
        h, w = final_result_color.shape[:2]
        input_resized = cv2.resize(input_img_color, (w, h))

        mask = numpy.zeros((h, w), dtype=numpy.float32)
        center_x, center_y = w // 2, h // 2
        cv2.circle(mask, (center_x, center_y), int(min(h, w) * 0.45), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=50, sigmaY=50)
        mask_3ch = cv2.merge([mask, mask, mask])

        final_blended = (final_result_color.astype(float) * mask_3ch +
                         input_resized.astype(float) * (1.0 - mask_3ch)).astype(numpy.uint8)

        # res finais
        utils.save_image(final_blended, "data/final_result_blended.jpg")
        utils.save_image(final_result_color, "data/final_result_lab.jpg")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Erro: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_image_path> <example_image_path>")
    else:
        main(sys.argv[1], sys.argv[2])
