import math
import biosppy
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import welch, lombscargle
import matplotlib.pyplot as plt

from pyhrv import tools


class HRV():

    def __init__(self, bvp, fs=35, distance=10):
        '''

        :param bvp: PPG信号
        :param frame_rate: 采样频率 次/秒
        :param distance: 波峰最小间隔
        '''
        super(HRV, self).__init__()
        assert isinstance(bvp, list) or isinstance(bvp, np.ndarray)
        self.bvp = bvp
        self.fs = fs
        self.bvp_peak = signal.find_peaks(bvp, distance=distance)
        # 计算波峰间隔 ms

        RRi = np.array(
            [(x[1] - x[0]) * 1000 / fs for x in zip(self.bvp_peak[0][:-1], self.bvp_peak[0][1:])])
        # 中值过滤噪声
        L = 51  # 中值滤波滑动窗长度
        PP1_flip = np.flip(RRi[2: int((L - 1) / 2 + 1)])
        PPlast_flip = np.flip(RRi[int(-((L - 1) / 2 + 1)):-2])
        RRi_flip = np.concatenate((PP1_flip, RRi, PPlast_flip))
        for i in range(len(RRi_flip) - L + 1):
            med_win = RRi_flip[i: i + L]
            medRR = np.median(med_win)
            if np.abs(RRi[i] - medRR) >= 150:
                RRi[i] = medRR
        self.RRi = RRi

    def get_hr(self):
        rri = self.RRi
        mhr = np.mean(60 / (rri / 1000.0))
        return mhr

    def get_pnn50(self):
        rri = self.RRi
        nn50 = sum(abs(np.diff(rri)) > 50)
        pnn50 = nn50 / len(rri) * 100
        return pnn50

    def plt_bvp(self):
        plt.plot(range(len(self.bvp)), self.bvp, 'r', label='ppg')
        plt.xlabel('frame')
        plt.ylabel('value')
        plt.legend(loc=4)
        plt.title('PPG')
        for ii in range(len(self.bvp_peak[0])):
            plt.plot(self.bvp_peak[0][ii], self.bvp[self.bvp_peak[0][ii]], '.b', markersize=3)
        plt.show()

    def time_domain(self):
        rri = self.RRi
        diff_rri = np.diff(rri)
        rmssd = np.sqrt(np.mean(diff_rri ** 2))
        sdnn = np.std(rri, ddof=1)  # make it calculates N-1
        sdsd = np.std(diff_rri, ddof=1)
        nn50 = sum(abs(np.diff(rri)) > 50)
        pnn50 = nn50 / len(rri) * 100
        mrri = np.mean(rri)
        mhr = np.mean(60 / (rri / 1000.0))

        return dict(
            zip(
                ["rmssd", "sdnn", "sdsd", "nn50", "pnn50", "mrri", "mhr"],
                [rmssd, sdnn, sdsd, nn50, pnn50, mrri, mhr],
            )
        )

    def frequency_domain(
            self,
            fbands={'ulf': None, 'vlf': (0.003, 0.04), 'lf': (0.04, 0.15), 'hf': (0.15, 0.4)},
            nfft=2 ** 8,
            ma_size=None
    ):
        rri = self.RRi
        t = np.cumsum(rri)
        t -= t[0]
        # Compute PSD according to the Lomb-Scargle method
        # Specify frequency grid
        frequencies = np.linspace(0, 0.41, nfft)
        # Compute angular frequencies
        a_frequencies = np.asarray(2 * np.pi / frequencies)
        powers = np.asarray(lombscargle(t, rri, a_frequencies, normalize=True))
        # Fix power = inf at f=0
        powers[0] = 0
        # Apply moving average filter
        if ma_size is not None:
            powers = biosppy.signals.tools.smoother(powers, size=ma_size)['signal']

        # Define metadata
        meta = biosppy.utils.ReturnTuple((nfft, ma_size,), ('lomb_nfft', 'lomb_ma'))
        powers = powers * 10 ** 6

        # Compute frequency parameters
        params, freq_i = _compute_parameters('lomb', frequencies, powers, fbands)

        # Complete output
        result = join_tuples(params, meta)
        tp = result['lomb_total']
        vlf = result['lomb_abs'][0]
        lf = result['lomb_abs'][1]
        hf = result['lomb_abs'][2]
        ratio_lf_hf = result['lomb_ratio']

        # return dict(
        #     zip(
        #         ["total_power", "vlf", "lf", "hf", "lf_hf"],
        #         [tp, vlf, lf, hf, ratio_lf_hf],
        #     )
        # )
        return dict(
            zip(
                ["vlf", "lf", "hf"],
                [vlf, lf, hf],
            )
        )

    def get_si(self):
        bin_width = 50
        min = np.min(self.RRi)
        max = np.max(self.RRi)
        MxDMn = max - min
        Mo = np.median(self.RRi)
        bins = math.ceil(MxDMn / bin_width)
        histogram = np.histogram(self.RRi, range=(min, max), bins=bins, density=True)
        AMo = np.max(histogram[0])
        return (AMo * 100) / (2 * Mo * MxDMn)

    def poincare(self):
        '''
        庞加莱特征
        '''
        rri = self.RRi
        x1 = np.asarray(rri[:-1])
        x2 = np.asarray(rri[1:])
        sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
        sd2 = np.std(np.add(x1, x2) / np.sqrt(2))
        return dict(
            zip(
                ["sd1", "sd2", "sd1_sd2", "s"],
                [sd1, sd2, sd1 / sd2, np.pi * sd1 * sd2],
            )
        )


def join_tuples(*args):
    """Joins multiple biosppy.biosppy.utils.ReturnTuple objects into one biosppy.biosppy.utils.ReturnTuple object.
    Docs:	https://pyhrv.readthedocs.io/en/latest/_pages/api/utils.html#join-tuples-join-tuples
    Parameters
    ----------
    tuples : list, array, biosppy.utils.ReturnTuple objects
        List or array containing biosppy.utils.ReturnTuple objects.
    Returns
    -------
    return_tuple : biosppy.biosppy.utils.ReturnTuple object
        biosppy.biosppy.utils.ReturnTuple object with the content of all input tuples joined together.
    Raises
    ------
    TypeError:
        If no input data is provided
    TypeError:
        If input data contains non-biosppy.biosppy.utils.ReturnTuple objects
    Notes
    ----
    ..	You can find the documentation for this function here:
        https://pyhrv.readthedocs.io/en/latest/_pages/api/utils.html#join-tuples-join-tuples
    """
    # Check input
    if args is None:
        raise TypeError("Please specify input data.")

    for i in args:
        if not isinstance(i, biosppy.utils.ReturnTuple):
            raise TypeError("The list of tuples contains non-biosppy.utils.ReturnTuple objects.")

    # Join tuples
    names = ()
    vals = ()

    for i in args:
        for key in i.keys():
            names = names + (key,)
            vals = vals + (i[key],)

    return biosppy.utils.ReturnTuple(vals, names)


def _create_interp_time(length, fs):
    time_resolution = 1000 / float(fs)
    return np.arange(0, length * time_resolution, time_resolution)


def _get_frequency_indices(freq, freq_bands):
    """Returns list of lists where each list contains all indices of the PSD frequencies within a frequency band.
    Parameters
    ----------
    freq : array
        Frequencies of the PSD.
    freq_bands : dict, optional
        Dictionary with frequency bands (tuples or list).
        Value format:	(lower_freq_band_boundary, upper_freq_band_boundary)
        Keys:	'ulf'	Ultra low frequency		(default: none) optional
                'vlf'	Very low frequency		(default: (0.003Hz, 0.04Hz))
                'lf'	Low frequency			(default: (0.04Hz - 0.15Hz))
                'hf'	High frequency			(default: (0.15Hz - 0.4Hz))
    Returns
    -------
    indices : list of lists
        Lists with all indices of PSD frequencies of each frequency band.
    """
    indices = []
    for key in freq_bands.keys():
        if freq_bands[key] is None:
            indices.append(None)
        else:
            indices.append(np.where((freq_bands[key][0] <= freq) & (freq <= freq_bands[key][1])))

    if indices[0] is None or len(indices) == 3:
        return None, indices[1][0], indices[2][0], indices[3][0]
    else:
        return indices[0][0], indices[1][0], indices[2][0], indices[3][0]


def _get_frequency_arrays(freq, ulf_i, vlf_i, lf_i, hf_i):
    """Returns arrays with all frequencies within each frequency band.
    Parameters
    ----------
    freq : array
        Frequencies of the PSD.
    ulf_i : array
        Indices of all frequencies within the ULF band.
    vlf_i : array
        Indices of all frequencies within the ULF band.
    lf_i : array
        Indices of all frequencies within the ULF band.
    hf_i : array
        Indices of all frequencies within the ULF band.
    Returns
    -------
    ulf_f : array
        Frequencies of the ULF band.
    vlf_f : array
        Frequencies of the VLF band.
    lf_f : array
        Frequencies of the LF band.
    hf_f : array
        Frequencies of the HF band.
    """
    ulf_f = np.asarray(freq[ulf_i]) if ulf_i is not None else None
    vlf_f = np.asarray(freq[vlf_i])
    lf_f = np.asarray(freq[lf_i])
    hf_f = np.asarray(freq[hf_i])
    return ulf_f, vlf_f, lf_f, hf_f


def _compute_parameters(method, frequencies, power, freq_bands):
    """Computes PSD HRV parameters from the PSD frequencies and powers.
    References: [Electrophysiology1996], [Basak2014]
    Docs:		https://pyhrv.readthedocs.io/en/latest/_pages/api/frequency.html#frequency-parameters
    Parameters
    ----------
    method : str
        Method identifier ('fft', 'ar', 'lomb')
    frequencies
        Series of frequencies of the power spectral density computation.
    power : array
        Series of power-values of the power spectral density computation.
    freq_indices : array
        Indices of the frequency samples within each frequency band.
    freq_bands : dict, optional
        Dictionary with frequency bands (tuples or list).
        Value format:	(lower_freq_band_boundary, upper_freq_band_boundary)
        Keys:	'ulf'	Ultra low frequency		(default: none) optional
                'vlf'	Very low frequency		(default: (0.003Hz, 0.04Hz))
                'lf'	Low frequency			(default: (0.04Hz - 0.15Hz))
                'hf'	High frequency			(default: (0.15Hz - 0.4Hz))
    Returns
    -------
    results : biosppy.utils.ReturnTuple object
        All results of the Lomb-Scargle PSD estimation (see list and keys below)
    Returned Parameters & Keys
    --------------------------
    (below, X = method identifier 'fft', 'ar' or 'lomb'
    ..	Peak frequencies of all frequency bands in [Hz] (key: 'X_peak')
    ..	Absolute powers of all frequency bands in [ms^2] (key: 'X_abs')
    ..	Relative powers of all frequency bands in [%] (key: 'X_rel')
    ..	Logarithmic powers of all frequency bands [-] (key: 'X_log')
    ..	Normalized powers of all frequency bands [-](key: 'X_norms')
    ..	LF/HF ratio [–] (key: 'X_ratio')
    ..	Total power over all frequency bands in [ms^] (key: 'X_total')
    Raises
    ------
    ValueError
        If parameter computation could not be made due to the lack of PSD samples ('nfft' too low)
    """
    # Compute frequency resolution
    df = (frequencies[1] - frequencies[0])

    # Get indices of freq values within the specified freq bands
    ulf_i, vlf_i, lf_i, hf_i = _get_frequency_indices(frequencies, freq_bands)
    ulf_f, vlf_f, lf_f, hf_f = _get_frequency_arrays(frequencies, ulf_i, vlf_i, lf_i, hf_i)

    # Absolute powers
    if freq_bands['ulf'] is not None:
        ulf_power = np.sum(power[ulf_i]) * df
    vlf_power = np.sum(power[vlf_i]) * df
    lf_power = np.sum(power[lf_i]) * df
    hf_power = np.sum(power[hf_i]) * df
    abs_powers = (vlf_power, lf_power, hf_power,) if freq_bands['ulf'] is None else (ulf_power, vlf_power, lf_power,
                                                                                     hf_power,)
    total_power = np.sum(abs_powers)

    # Peak frequencies
    if freq_bands['ulf'] is not None:
        ulf_peak = ulf_f[np.argmax(power[ulf_i])]

    # Compute Peak values and catch exception caused if the number of PSD samples is too low
    try:
        vlf_peak = vlf_f[np.argmax(power[vlf_i])]
        lf_peak = lf_f[np.argmax(power[lf_i])]
        hf_peak = hf_f[np.argmax(power[hf_i])]
        peaks = (vlf_peak, lf_peak, hf_peak,) if freq_bands['ulf'] is None else (ulf_peak, vlf_peak, lf_peak, hf_peak,)
    except ValueError as e:
        if 'argmax of an empty sequence' in str(e):
            raise ValueError("'nfft' is too low: not enough PSD samples to compute the frequency parameters. Try to "
                             "increase 'nfft' to avoid this error.")

    # Relative, logarithmic powers & LF/HF ratio
    rels = tuple([float(x) / total_power * 100 for x in abs_powers])
    logs = tuple([float(np.log(x)) for x in abs_powers])
    ratio = float(lf_power) / hf_power

    # Normalized powers
    norms = tuple([100 * x / (lf_power + hf_power) for x in [lf_power, hf_power]])

    # Prepare parameters for plot
    args = (freq_bands, peaks, abs_powers, rels, logs, norms, ratio, total_power)
    names = (
        '%s_bands' % method, '%s_peak' % method, '%s_abs' % method,
        '%s_rel' % method, '%s_log' % method, '%s_norm' % method,
        '%s_ratio' % method, '%s_total' % method)

    # Output
    params = biosppy.utils.ReturnTuple(args, names)
    freq_i = biosppy.utils.ReturnTuple((ulf_i, vlf_i, lf_i, hf_i), ('ulf', 'vlf', 'lf', 'hf'))
    return params, freq_i


if __name__ == '__main__':
    bvp_ = pd.read_csv(r"E:\dataset\ubfc-phys\processed\train\s10\T1\rppg_s10_T1.csv")
    bvp_ = bvp_.to_numpy()
    bvp_ = bvp_[:, -1]
    # hrv = HRV(bvp_[175:-175], 64, 30)
    hrv = HRV(bvp_[175:-175])
    print(hrv.time_domain())
    print(hrv.frequency_domain())
    hrv.plt_bvp()
