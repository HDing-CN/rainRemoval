import os

import cv2
import numpy as np


def read_video(dir_name, mode='RGB'):
    cap = cv2.VideoCapture(dir_name)
    frame_nums = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('frame_nums:', frame_nums)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('size: ', size)
    state, frame = cap.read()
    frame_out = []
    while state:
        if mode == 'HSL':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_out.append(frame)
        cv2.waitKey(1)
        state, frame = cap.read()
    cap.release()
    frame_out = np.asarray(frame_out, dtype=np.uint8)
    return frame_out

def read_all_img(dir_name):
    img_paths = os.listdir(dir_name)
    img_paths = [i for i in img_paths if i.split('.')[1] == 'jpg']
    img_paths.sort(key=lambda x: int(x.split('.')[0]))
    out = []
    for file in img_paths:
        img = cv2.imread(os.path.join(dir_name, file), 1)
        # img = cv2.medianBlur(img, 3)
        out.append(img)
    print('frame_nums:', len(out))
    print('size:', out[0].shape)
    out = np.asarray(out, dtype=np.uint8)
    return out
