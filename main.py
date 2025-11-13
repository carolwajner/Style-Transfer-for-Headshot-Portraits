import sys
import utils

def main(arg: str):
    try:
        image = utils.read_image(arg)

        utils.save_image(image, "data/output.jpg")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
    else:
        main(sys.argv[1])