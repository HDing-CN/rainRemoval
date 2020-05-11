import numpy as np
import cv2
import os
from itertools import cycle
from Converter import play_video, convert_images_to_video, convert_npy_to_images, numOfDigits
from ReadFile import read_video


# frame_path = "results_from_npy/"  # 图片的文件夹路径
# img_list = []
# for i in range(1046):
#     img_list.append(cv2.imread(frame_path + 'result_' + ((4 - numOfDigits(i + 1)) * '0') + str(i + 1) + '.jpg'))
# play_video(img_list)


# convert_images_to_video('./rgb_video/crossing/', 'gt_car.avi', 20.0)

video = read_video('./rgb_video/rgb_rain.mp4')
for i in range(len(video)):
    cv2.imwrite('./rgb_video/rgb_rain/rgb_rain_' + ((4 - numOfDigits(i + 1)) * '0') + str(i + 1) + '.jpg', video[i])
# convert_npy_to_images('all_res_car.npy', 'results_from_car_npy/')