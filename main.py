from scipy.io import loadmat

from read_file import *
from converter import *
from derain import *
from rpca_admm import *
from rpca import TRPCA

def do_derain():
    m = loadmat('./man/man.mat')
    video = trans(m["input"])
    print (video.shape)

    back, fore, noise = bfr(video)
    np.save("./man/buffer_test/back.npy", back)
    np.save("./man/buffer_test/fore.npy", fore)
    np.save("./man/buffer_test/noise.npy", noise)

do_derain()
# m = loadmat('./man/man.mat')
# video = trans(m["input"])
# print (video.shape)
# img = video[:, 100, :, 0]
# cv2.imwrite("test.jpg", img)
# h = rpcaADMM(img)
# cv2.imwrite("test1.jpg", h['X1_admm'])
# cv2.imwrite("test2.jpg", h['X2_admm'])
# cv2.imwrite("test3.jpg", h['X3_admm'])
# cv2.imwrite("test4.jpg", h['X1_admm'] + h['X2_admm'])
#
# trpca = TRPCA()
# hh, nn = trpca.ADMM(img)
# cv2.imwrite("nagehao.jpg", hh)

# denoise = np.load("./man/buffer_test/denoise.npy")
# cv2.imwrite("shibushiwoxiangdenayang.jpg", denoise[4])

# denoise = np.load("./man/buffer_test/denoise.npy")
# for i in range(240):
#     cv2.imwrite(generate_file_name("./man/buffer_test/", i), denoise[:, i, :, :])