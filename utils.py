import cv2


def read_image(path: str) -> cv2.typing.MatLike:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError("The image file was not found.")
    return image


def save_image(image: cv2.typing.MatLike, path: str) -> None:
    cv2.imwrite(path, image)
