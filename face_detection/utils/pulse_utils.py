import numpy as np

def moving_avg(signal, w_s):
    '''
    卷积操作，求信号在w_s长度卷积上的平均
    :param signal: 源信号
    :param w_s: 卷积长度
    :return: 平均卷积处理后的信号
    '''
    ones = np.ones(w_s) / w_s
    moving_avg = np.convolve(signal, ones, 'valid')
    return moving_avg