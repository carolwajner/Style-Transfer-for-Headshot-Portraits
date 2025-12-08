import sys
import utils
import dense_correspondence
import local_contrast_transfer
import background

def main(arg1: str, arg2: str):
    try:
        input_img = utils.read_image(arg1)
        example_img = utils.read_image(arg2)

        print("Realizando morph...")
        morphed_img = dense_correspondence.dense_morph(input_img, example_img)
        utils.save_image(morphed_img, "data/morphed.jpg")

        print("Extraindo background...")
        mask_in = background.get_segmentation_mask(input_img)
        mask_ex = background.get_segmentation_mask(example_img)
        clean_bg_ex = background.inpainting_background_extraction(example_img, mask_ex)

        print("Style transfer...")
        face_on_black = local_contrast_transfer.apply_local_contrast_and_blend(
            input_img=input_img,
            morphed_img=morphed_img,
            input_img_color=input_img,
            segmentation_mask=mask_in,
            clean_background=clean_bg_ex
        )

        print("Mistura de backgrounds...")
        final_result = background.replace_background(
            foreground_img=face_on_black,
            background_img=clean_bg_ex,
            mask=mask_in
        )

        utils.save_image(final_result, "results/final_result_blended.jpg")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Erro: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_image_path> <example_image_path>")
    else:
        main(sys.argv[1], sys.argv[2])
