import numpy as np
from sklearn.decomposition import FastICA
from utils.filter import butter_bandpass_filter, detrend


class ICA():
    def __init__(self, fs):
        super(ICA, self).__init__()
        self.lowcut = 0.7
        self.highcut = 2.5
        self.fs = fs

    def smooth(self, x, window_len=11, window='hanning'):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

        output:
            the smoothed signal

        example:

            t=linspace(-2,2,0.1)
            x=sin(t)+randn(len(t))*0.1
            y=smooth(x)

        see also:

            numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
            scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise ValueError('Smooth only accepts 1 dimension arrays.')

        if x.size < window_len:
            raise ValueError('Input vector needs to be bigger than window size.')

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    def get_pulse(self, mean_rgb):
        ica = FastICA(whiten=True)
        r_mean, g_mean, b_mean = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        B_bp = butter_bandpass_filter(b_mean, self.fs, self.lowcut, self.highcut, order=6)
        G_bp = butter_bandpass_filter(g_mean, self.fs, self.lowcut, self.highcut, order=6)
        R_bp = butter_bandpass_filter(r_mean, self.fs, self.lowcut, self.highcut, order=6)
        S = np.c_[B_bp, G_bp, R_bp]
        S /= S.std(axis=0)
        X = S
        S_ = ica.fit(X).transform(X)

        y = self.smooth(S_[:, 0], window_len=11, window='flat')
        y1 = self.smooth(S_[:, 1], window_len=11, window='flat')
        y2 = self.smooth(S_[:, 2], window_len=11, window='flat')

        raw = np.fft.rfft(y)
        raw1 = np.fft.rfft(y1)
        raw2 = np.fft.rfft(y2)

        fft = np.abs(raw)
        fft1 = np.abs(raw1)
        fft2 = np.abs(raw2)

        freqfft = [np.argmax(fft), np.argmax(fft1), np.argmax(fft2)]

        return S_[:, np.argmax(freqfft)]