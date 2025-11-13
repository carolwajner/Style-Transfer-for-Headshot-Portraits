import cv2

def read_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError("The image file was not found.")
    return image

def save_image(image: cv2.typing.MatLike, path: str):
    cv2.imwrite(path, image)
    


