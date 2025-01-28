import random

import behavioral_cloning_utils as utils
import os
import cv2


def augment(data_dir, index):
    """
    Run random augmentations on every image in data_dir.
    """
    for file in os.listdir(data_dir):
        if not file.endswith(".png"):
            continue

        img, steer = utils.load_image(data_dir, file, include_left_right=False, include_augmented=False)

        if img is None:
            continue

        if index == 1:
            img, steer = utils.flip_horizontal(img, steer)
        elif index == 2:
            img, steer = utils.random_translate(img, steer)
        elif index == 3:
            img = utils.random_shadow(img)
        elif index == 4:
            img = utils.maximum_contrast(img)
        elif index == 5:
            img = utils.to_grayscale(img)
        elif index == 6:
            img = utils.invert_color(img)
        elif index == 7:
            img = utils.add_random_rectangle(img)
        elif index == 8:
            img = utils.change_saturation(img)

        filename = f"{file[0:5]}_s{steer:+.3f}_augment_{index}"
        filename = filename.replace(".", ",")
        filename += ".png"

        cv2.imwrite(os.path.join(data_dir, filename), img)


def main():
    data_dir = r'C:\Users\A42893\Documents\FE\Workshop\test'
    num_augmentations = 8

    for i in range(1, num_augmentations + 1):
        print(f"Augmentation {i}...")
        augment(data_dir, i)


if __name__ == '__main__':
    main()
