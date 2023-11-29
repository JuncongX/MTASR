from scipy.signal import butter, lfilter
from scipy import signal
from scipy import sparse
import numpy as np
import math
from scipy.signal.windows import hann


# 巴特沃斯带通滤波
def butter_bandpass_filter(data, fs, lowcut=0.7, highcut=3.5, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

#  巴特沃斯高通滤波
def butter_highpass_filter(data, fs, lowcut=0.7, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='high')
    y = lfilter(b, a, data)
    return y

#  巴特沃斯低通滤波
def butter_lowpass_filter(data, fs, lowcut=0.7, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    y = lfilter(b, a, data)
    return y


def detrend(X, detLambda=10):
    """
    desc: get rid of a randomness trend might deal with sudden increase trend coming from head movements

    args:
        - X::[array<float>]
            signal
    ret:
        - detrendedX::[array<float>]
            detrended signal
    """
    # Smoothness prior approach as in the paper appendix:
    # "An advanced detrending method with application to HRV analysis"
    # by Tarvainen, Ranta-aho and Karjaalainen
    t = X.shape[0]
    l = t / detLambda  # lambda
    I = np.identity(t)
    D2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(t - 2, t)).toarray()  # this works better than spdiags in python
    detrendedX = (I - np.linalg.inv(I + l ** 2 * (np.transpose(D2).dot(D2)))).dot(X)
    return detrendedX


def moving_avg(signal, w_s):
    ones = np.ones(w_s) / w_s
    moving_avg = np.convolve(signal, ones, 'valid')
    return moving_avg
