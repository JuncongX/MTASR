import numpy as np
import math
import cv2

from utils.filter import butter_bandpass_filter, detrend


class CHROM():
    """ This method is described in the following paper:
        "Remote heart rate variability for emotional state monitoring"
        by Y. Benezeth, P. Li, R. Macwan, K. Nakamura, R. Gomez, F. Yang
    """

    def __init__(self, FS, FN, WinSec=1.6, is_x_preprocess=False):
        super(CHROM, self).__init__()
        self.is_x_preprocess = is_x_preprocess
        self.FS = FS  # 采样频率（摄像头帧率）
        self.FN = FN  # 总帧数
        self.WinL = math.ceil(WinSec * FS)  # 计算
        if self.WinL % 2 != 0:  # 强制WinL为偶数
            self.WinL = self.WinL + 1
        _NWin = (FN - self.WinL / 2) / (self.WinL / 2)
        self.NWin = math.floor(_NWin)
        self.WinP = math.ceil(_NWin) - self.NWin
        self.S = []
        self.WinS = 0  # Window Start Index
        self.WinM = self.WinS + int(self.WinL / 2)  # Window Middle Index
        self.WinE = self.WinS + self.WinL  # Window End Index

    def apply(self, X):
        if not self.is_x_preprocess:
            X = [skin_segment(x.astype(np.uint8)) for x in X]
            X = np.sum(np.sum(X, axis=1), axis=1) / (np.sum(np.sum((np.array(X) > 0), axis=1), axis=1) + 1e-15)
        X = np.array(X, dtype=np.float64)


        # print(X[self.WinS:self.WinE])
        RGBBase = np.average(X, axis=0)
        RGBNorm = X / (RGBBase + 1e-15) - 1
        # image channels should be RGB
        # calculation of new X and Y
        Xcomp = 3 * RGBNorm[:, 0] - 2 * RGBNorm[:, 1]
        Ycomp = (1.5 * RGBNorm[:, 0]) + RGBNorm[:, 1] - (1.5 * RGBNorm[:, 2])

        Xf = butter_bandpass_filter(Xcomp, self.FS, order=3)
        Yf = butter_bandpass_filter(Ycomp, self.FS, order=3)

        # standard deviations
        sX = np.std(Xf)
        sY = np.std(Yf)

        alpha = sX / (sY + 1e-15)

        # -- rPPG signal
        S = Xf - alpha * Yf

        return S


def skin_segment(bgr_image):
    ycrcb_image = None
    try:
        ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)
    except Exception as e:
        print(e)
        print(bgr_image.shape)
    H, W, C = ycrcb_image.shape
    mask = np.zeros((H, W), dtype="uint8")
    cb = ycrcb_image[:, :, 2]
    cr = ycrcb_image[:, :, 1]
    y = ycrcb_image[:, :, 0]
    cb_index = np.logical_and(cb > 77, cb < 127)
    cr_index = np.logical_and(cr > 137, cr < 177)
    y_index = np.logical_and(y > 80, y < 255)
    mask[np.logical_and(cb_index, cr_index, y_index)] = 1
    SKIN_ROI = cv2.add(bgr_image, np.zeros(np.shape(bgr_image), dtype=np.uint8), mask=mask)
    return SKIN_ROI

