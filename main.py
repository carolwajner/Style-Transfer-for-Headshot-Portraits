import sys
import utils
import face_correspondence
import morphing
import dense_flow

def main(arg1: str, arg2: str):
    try:
        input_img = utils.read_image(arg1)
        example_img = utils.read_image(arg2)

        # Alinhamento Afim
        points_input, points_example = face_correspondence._detect_landmarks(input_img, example_img)

        affine_matrix = face_correspondence._affine_transform(points_example, points_input)
        aligned_example = utils.cv2.warpAffine(
            example_img,
            affine_matrix,
            (input_img.shape[1], input_img.shape[0]),
            borderMode=utils.cv2.BORDER_REPLICATE
        )

        # Morphing de Beier and Neely
        points_aligned_example, _ = face_correspondence._detect_landmarks(aligned_example, input_img)
        morphed_output = morphing.beier_neely(aligned_example, points_aligned_example, points_input)

        # Optical Flow (SIFT)
        final_output = dense_flow.apply_adjustment(input_img, morphed_output)

        utils.save_image(final_output, "data/output4.jpg")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Erro: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_image_path> <example_image_path>")
    else:
        main(sys.argv[1], sys.argv[2])