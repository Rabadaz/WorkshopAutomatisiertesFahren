import os
import cv2


def main():
    save_dir = r'D:\Fabio\Dokumente\FH\5_Semester\ATF\track_all'
    data_dirs = [r'D:\Fabio\Dokumente\FH\5_Semester\ATF\track_3', r'D:\Fabio\Dokumente\FH\5_Semester\ATF\track_4', r'D:\Fabio\Dokumente\FH\5_Semester\ATF\track_5']
    idx = 0
    new_img = 0

    for data_dir in data_dirs:
        files = os.listdir(data_dir)

        for file in files:
            if not file.endswith(".png"):
                continue

            img = cv2.imread(os.path.join(data_dir, file))
            filename = f"{idx:05d}{file[5:]}"
            cv2.imwrite(os.path.join(save_dir, filename), img)
            new_img = new_img + 1
            if new_img == 3:
                idx = idx + 1
                new_img = 0


if __name__ == '__main__':
    main()
