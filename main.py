import sys
import utils
import dense_correspondence


def main(arg1: str, arg2: str):
    try:
        input_img = utils.read_image(arg1)
        example_img = utils.read_image(arg2)

        output = dense_correspondence.dense_morph(input_img, example_img)

        utils.save_image(output, "data/output.jpg")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Erro: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_image_path> <example_image_path>")
    else:
        main(sys.argv[1], sys.argv[2])
