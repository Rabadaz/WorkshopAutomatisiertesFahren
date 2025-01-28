import cv2
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file, include_left_right=False, include_augmented=False):
    """
    Load RGB image from file and adjust steering angle accordingly

    @param include_left_right:  only applies if file path points to an image from the left/right camera (name contains
                                "left"/"right")
                                if true, function returns (stored image, adjusted steering angle from filename)
                                otherwise function returns (None, 0)
    @param include_augmented:   only applies if file path points to an augmented image (name contains "augmented")
                                if true, function returns (stored image, steering angle from filename)
                                otherwise function returns (None, 0)
    """

    img = cv2.imread(os.path.join(data_dir, image_file))
    steer = extract_steering_from_filename(image_file)

    if not include_left_right and ("left" in image_file or "right" in image_file):
        return None, 0

    if not include_augmented and "augment" in image_file:
        return None, 0

    if include_left_right:
        if "left" in image_file:
            steer = steer + 0.2
        elif "right" in image_file:
            steer = steer - 0.2

    return img, steer


def extract_steering_from_filename(file):
    """
    Extract the recorded steering angle from the filename.
    e.g.: "00137_s+0,066_center.png" -> steering angle = 0.066
    """
    return float(file[7:13].replace(',', '.'))


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[12:78, 50:250, :]  # remove the sky and the car front


def bgr2yuv(image):
    """
    Convert image from RGB to YUV (this is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = bgr2yuv(image)
    return image


def flip_horizontal(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle

    return image, steering_angle


def flip_vertical(image, steering_angle):
    image = cv2.flip(image, 0)
    steering_angle = -steering_angle

    return image, steering_angle


def random_translate(image, steering_angle, h_range=50, v_range=5):
    """
    Randomly shift the image vertically and horizontally (translation).
    """
    height, width = image.shape[:2]

    t_h = random.choice([-h_range, h_range])
    t_v = random.choice([-v_range, v_range])

    T = np.float32([[1, 0, t_h], [0, 1, t_v]])

    image = cv2.warpAffine(image, T, (width, height))
    steering_angle = steering_angle + t_h * 0.015

    return image, steering_angle


def random_shadow(image):
    """
    Generate and add random shadow
    """
    height, width, _ = image.shape
    brightness_factor = random.uniform(0.4, 0.6)

    mask_pil = Image.new("L", (width, height), "white")
    draw = ImageDraw.Draw(mask_pil)
    left_start = random.randrange(0, height)
    right_start = random.randrange(0, height)
    draw.polygon([(0, left_start), (width, right_start), (width, height), (0, height)], fill="black")
    mask = np.array(mask_pil) / 255

    if random.choice([True, False]):
        mask = 1 - mask

    mask[mask == 0] = brightness_factor

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:, :, 2] = image[:, :, 2] * mask
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def maximum_contrast(img):
    B, G, R = cv2.split(img)
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    out = cv2.merge((B, G, R))
    return out


def to_grayscale(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image


def invert_color(img):
    inv = cv2.bitwise_not(img)
    return inv


def add_random_rectangle(img):
    height, width, _ = img.shape

    # Random size for the rectangle
    rect_width = random.randint(10, width // 2)
    rect_height = random.randint(10, height // 2)

    # Random position for the rectangle (top-left corner)
    top_left_x = random.randint(0, width - rect_width)
    top_left_y = random.randint(0, height - rect_height)

    # Calculate bottom-right corner based on random size
    bottom_right_x = top_left_x + rect_width
    bottom_right_y = top_left_y + rect_height

    # Random color for the rectangle in BGR format
    color = [random.randint(0, 255) for _ in range(3)]

    # Draw the rectangle on the image
    modified_image = cv2.rectangle(img.copy(), (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color,
                                   -1)

    return modified_image


def change_saturation(img, saturation_factor=1.5):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Adjust the saturation channel
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)

    # Convert the image back from HSV to BGR
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return saturated_image


def edge_detection(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 10, 100)

    return edges


def test():
    image = cv2.imread(r'C:\Users\A42893\Documents\FE\Workshop\test\00044_s-0,025_center.png')

    image = edge_detection(image)
    cv2.imshow("shadow", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    test()
