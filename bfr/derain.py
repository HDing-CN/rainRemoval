import numpy as np
from rpca import TRPCA
from rpca_admm import *


def temporal_filter(video, view='tw'):
    N, H, W, C = video.shape
    denoise_data = np.zeros(video.shape)
    noise_data = np.zeros(video.shape)
    trpca = TRPCA()
    for k in range(C):
        print('Processing the ' + str(k + 1) + '-th channel')
        if view == 'tw':
            for i in range(H):
                # if (i % 10 == 0):
                print('Processing the ' + str(i + 1) + '-th frame on the ' + str(k + 1) + '-th channel')
                img = video[:, i, :, k]
                # denoise, noise = trpca.ADMM(img)
                # denoise_data[:, i, :, k] = denoise
                # noise_data[:, i, :, k] = noise

                h = rpcaADMM(img)
                denoise_data[:, i, :, k] = h['X3_admm']
                noise_data[:, i, :, k] = h['X1_admm'] + h['X2_admm']


        elif view == 'th':
            for i in range(W):
                # print('Processing the ' + str(i + 1) + '-th frame on the ' + str(k + 1) + '-th channel')
                img = video[:, :, i, k]
                denoise, noise = trpca.ADMM(img)
                denoise_data[:, :, i, k] = denoise
                noise_data[:, :, i, k] = noise
        elif view == 'wh':
            for i in range(N):
                # print('Processing the ' + str(i + 1) + '-th frame on the ' + str(k + 1) + '-th channel')
                img = video[i, :, :, k]
                denoise, noise = trpca.ADMM(img)
                denoise_data[i, :, :, k] = denoise
                noise_data[i, :, :, k] = noise
    return denoise_data, noise_data


def bfr(video):
    N, H, W, C = video.shape
    background_data = np.zeros(video.shape)
    foreground_data = np.zeros(video.shape)
    noise_data = np.zeros(video.shape)

    for k in range(C):
        print('Processing the ' + str(k + 1) + '-th channel')
        for i in range(H):
            # if (i % 10 == 0):
            print('Processing the ' + str(i + 1) + '-th frame on the ' + str(k + 1) + '-th channel')
            img = video[:, i, :, k]
            # denoise, noise = trpca.ADMM(img)
            # denoise_data[:, i, :, k] = denoise
            # noise_data[:, i, :, k] = noise

            h = rpcaADMM(img)
            background_data[:, i, :, k] = h['X3_admm']
            foreground_data[:, i, :, k] = h['X2_admm']
            noise_data[:, i, :, k] = h['X1_admm']

    return background_data, foreground_data, noise_data


def derain_with_buffer(video, buffer, view='tw'):
    N, H, W, C = video.shape
    denoise_data = np.zeros(video.shape)
    noise_data = np.zeros(video.shape)
    now = 0
    while now + buffer <= N:
        print("Buffer: [" + str(now) + ", " + str(now + buffer) + ")")
        temp_denoise, temp_noise = temporal_filter(video[now: now + buffer], view)
        if now + buffer < N:
            denoise_data[now] = temp_denoise[0]
            noise_data[now] = temp_noise[0]
        elif now + buffer == N:
            denoise_data[now:] = temp_denoise
            noise_data[now:] = temp_noise
        now = now + 1
    # if now < N:
    #     denoise_data[now:], noise_data[now:] = temporal_filter(video[now:], view)
    return denoise_data, noise_data
