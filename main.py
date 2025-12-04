import sys
import utils
import dense_correspondence
import local_contrast_transfer


def main(arg1: str, arg2: str):
    try:
        input_img = utils.read_image(arg1)
        example_img = utils.read_image(arg2)

        # backup da imagem pq deu errado sem isso e com isso tbm da, mass fica melhor
        # o problema é q a camiseta do rapaz fica azul
        input_img_color = input_img.copy()

        # geometria e deformação
        output = dense_correspondence.dense_morph(input_img, example_img)

        final_blended, final_result_color = (
            local_contrast_transfer.apply_local_contrast_and_blend(
                input_img,
                output,
                input_img_color,
            )
        )

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
