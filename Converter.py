import numpy as np
import cv2
from itertools import cycle
import os
from os.path import isfile, join


def convert_single_channel(video, view):
    print('start converting to new viewing angle only on single color channel')
    video = np.asarray(video)
    N, H, W, C = video.shape
    new_video = []
    for k in range(C):
        if view == 'tw':
            for i in range(H):
                img = video[:, i, :, k]
                new_video.append(img)
        elif view == 'th':
            for i in range(W):
                img = video[:, :, i, k]
                new_video.append(img)

    print('the shape of new video is: ' + str(np.array(new_video).shape))
    print('----------------------')
    return new_video


def convert_RGB_channel(video, view):
    print('start converting to new viewing angle including all 3 RGB channels')
    video = np.asarray(video)
    N, H, W, C = video.shape
    new_video = []
    if view == 'tw':
        for i in range(H):
            img = video[:, i, :, :]
            new_video.append(img)
    elif view == 'th':
        for i in range(W):
            img = video[:, :, i, :]
            new_video.append(img)

    print('the shape of new video is: ' + str(np.array(new_video).shape))
    print('----------------------')
    return new_video


def play_video(img_list):
    img_iter = cycle(img_list)

    key = 0
    while key & 0xFF != 27:
        cv2.imshow('window title', next(img_iter))
        key = cv2.waitKey(60)  # 1000为间隔1000毫秒 cv2.waitKey()参数不为零的时候则可以和循环结合产生动态画面


def convert_images_to_video(path_in, path_out, fps):
    frame_array = []
    files = [f for f in os.listdir(path_in) if isfile(join(path_in, f))]
    # for sorting the file names properly
    files.sort()
    print (files)
    for i in range(len(files)):
        if files[i][-3:] == 'jpg':
            filename = path_in + files[i]
            # reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)

            # inserting the frames into an image array
            frame_array.append(img)
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def numOfDigits(x):
    res = 0
    while x > 0:
        res += 1
        x = x // 10
    return res


def convert_npy_to_images(npy_file, path_out):
    npy_data = np.load(npy_file)
    for i in range(npy_data.shape[0]):
        print(i + 1)
        cv2.imwrite(path_out + 'result_' + ((4 - numOfDigits(i + 1)) * '0') + str(i + 1) + '.jpg', npy_data[i])
