import sys
import utils
import dense_correspondence
import local_contrast_transfer
import postrocessing
import background


def main(arg1: str, arg2: str):
    try:
        input_img = utils.read_image(arg1)
        example_img = utils.read_image(arg2)

        # backup da imagem pq deu errado sem isso e com isso tbm da, mass fica melhor
        # o problema é q a camiseta do rapaz fica azul
        input_img_color = input_img.copy()

        # geometria e deformação
        output = dense_correspondence.dense_morph(input_img, example_img)

        utils.save_image(output, "data/morphed.jpg")

        final_blended, final_result_color = (
            local_contrast_transfer.apply_local_contrast_and_blend(
                input_img,
                output,
                input_img_color,
            )
        )

        original_landmarks, example_landmarks = dense_correspondence.detect_landmarks(final_blended, example_img)

        #final_blended = postrocessing.transfer_eye_highlights(final_blended, example_img,  original_landmarks, example_landmarks)

        final_blended = background.replace_background(final_blended, example_img,  original_landmarks, example_landmarks )

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
