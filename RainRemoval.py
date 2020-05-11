from ReadFile import read_all_img, read_video
from Converter import convert_single_channel, convert_RGB_channel, play_video
from sklearn.decomposition import PCA
import cv2
from RPCA import TRPCA
import numpy as np

rgb_pic_path = './rgb_video/crossing/'
rgb_rain_path = './rgb_video/rgb_rain.mp4'

# video = read_video(rgb_rain_path)
video = read_all_img(rgb_pic_path)
# play_video(video)
T, H, W, C = np.array(video).shape

# video = convert_RGB_channel(video, 'tw')
video = convert_single_channel(video, 'tw')
# play_video(video)

print(T, H, W, C)
num_of_height_frame = len(video) // 3
trpca = TRPCA()
all_res = np.zeros((T, H, W, C))
for k in range(num_of_height_frame):
    res = np.zeros((T, W, C))
    for i in range(C):
        print('Processing the ' + str(k + 1) + '-th frame on the ' + str(i + 1) + '-th channel')
        img = video[k + i * num_of_height_frame]
        denoise, noise = trpca.ADMM(img)
        res[:,:,i] = denoise
    cv2.imwrite('results_car/result_' + str(k + 1) + '.jpg', res)
    all_res[:,k,:,:] = res
    print('The ' + str(k + 1) + '-th frame finished.')

np.save('all_res_car.npy', all_res)

# cv2.imwrite('test/test_res.jpg', res)
# cv2.imwrite('test/test_gt.jpg', gt)

# img = video[200]
# print(len(video))
# print(img.shape)
# cv2.imwrite('test/test1.jpg', img)

# pca = PCA()
# tmp = pca.fit_transform(video[0])
# res = pca.inverse_transform(tmp)
# cv2.imshow('test', res)
# cv2.waitKey(0)
