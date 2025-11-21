import sys
import utils
import face_correspondence
import morphing


def main(arg1: str, arg2: str):
    try:
        input_img = utils.read_image(arg1)
        example_img = utils.read_image(arg2)

        points_input, points_example = face_correspondence._detect_landmarks(input_img, example_img)

        affine_matrix = face_correspondence._affine_transform(points_example, points_input)
        aligned_example = utils.cv2.warpAffine(example_img, affine_matrix, (input_img.shape[1], input_img.shape[0]))

        points_aligned_example, _ = face_correspondence._detect_landmarks(aligned_example, input_img)
        output = morphing.beier_neely(aligned_example, points_aligned_example, points_input)

        utils.save_image(output, "data/output2.jpg")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Erro: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_image_path> <example_image_path>")
    else:
        main(sys.argv[1], sys.argv[2])