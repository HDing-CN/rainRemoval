import cv2
import os
import time
import numpy as np


def read_video(dir_name):
    print('start reading video: ' + dir_name)
    cap = cv2.VideoCapture(dir_name)
    frame_nums = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('frame_nums:', frame_nums)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    state, frame = cap.read()
    frame_out = []
    while state:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # ???
        frame_out.append(frame)
        cv2.waitKey(1)
        state, frame = cap.read()
    cap.release()

    print('output shape is: ' + str(np.asarray(frame_out).shape))
    print('----------------------')
    return frame_out


def read_all_img(dir_name):
    print('start reading images in: ' + dir_name)
    img_paths = os.listdir(dir_name)
    img_paths = [i for i in img_paths if i.split('.')[1] == 'jpg']
    img_paths.sort(key=lambda x: int(x.split('.')[0]))
    out = []
    for file in img_paths:
        img = cv2.imread(os.path.join(dir_name, file), 1)
        # img = cv2.medianBlur(img, 3)
        out.append(img)
    print('frame_nums:', len(out))
    out = np.asarray(out, dtype=np.uint8)
    print('output shape is: ' + str(np.asarray(out).shape))
    print('----------------------')
    return out
