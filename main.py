import sys
import utils


def main(arg1: str, arg2: str):
    try:
        input = utils.read_image(arg1)
        example = utils.read_image(arg2)

        utils.save_image(input, "data/output.jpg")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_image_path> <example_image_path>")
    else:
        main(sys.argv[1], sys.argv[2])
