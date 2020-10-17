import os
from os.path import isfile, join

import cv2
import numpy as np


# H W C N --> N H W C
def trans(data):
    data = np.swapaxes(data, 0, 3)  # 3 1 2 0
    data = np.swapaxes(data, 1, 3)  # 3 0 2 1
    data = np.swapaxes(data, 2, 3)  # 3 0 1 2
    data = data[:, :, :, [2, 1, 0]]
    return data


def numOfDigits(x):
    res = 0
    while x > 0:
        res += 1
        x = x // 10
    return res


def generate_file_name(path, i):
    return path + ((4 - numOfDigits(i + 1)) * '0') + str(i + 1) + '.jpg'


def convert_npy_to_images(npy_file, path_out):
    npy_data = np.load(npy_file)
    for i in range(npy_data.shape[0]):
        print(i + 1)
        cv2.imwrite(generate_file_name(path_out, i), npy_data[i])


def convert_images_to_video(path_in, path_out, fps):
    frame_array = []
    files = [f for f in os.listdir(path_in) if isfile(join(path_in, f))]
    # for sorting the file names properly
    files.sort()
    print(files)
    for i in range(len(files)):
        if files[i][-3:] == 'jpg':
            filename = path_in + files[i]
            # reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)

            # inserting the frames into an image array
            frame_array.append(img)
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
