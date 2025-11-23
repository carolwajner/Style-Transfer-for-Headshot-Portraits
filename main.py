import sys
import utils
import dense_correspondence
import local_contrast_transfer

import cv2


def main(arg1: str, arg2: str):
    try:
        input_img = utils.read_image(arg1)
        example_img = utils.read_image(arg2)

        output = dense_correspondence.dense_morph(input_img, example_img)

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2GRAY)

        laplacian_stack_input, residual_input = (
            local_contrast_transfer.create_laplacian_stacks(input_img, 7)
        )
        laplacian_stack_output, residual_output = (
            local_contrast_transfer.create_laplacian_stacks(example_img, 7)
        )

        utils.save_image(output, "data/output2.jpg")
        utils.save_image(residual_output, "data/residual_output.jpg")
        for i, layer in enumerate(laplacian_stack_output):
            layer = cv2.convertScaleAbs(layer * 5, alpha=1.0, beta=128)
            utils.save_image(layer, f"data/layer_{i}_output.jpg")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Erro: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_image_path> <example_image_path>")
    else:
        main(sys.argv[1], sys.argv[2])
