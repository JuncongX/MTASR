# Exploiting Spatial Redundancy of Image Sensor for Motion Robust rPPG
# 脉冲一致性
import matlab
import matlab.engine
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import signal
from utils.HRV import HRV
from utils.filter import butter_bandpass_filter, moving_avg, detrend
from scipy.interpolate import interp1d

import yaml

eng = matlab.engine.start_matlab()

yaml_file = "../setting.yaml"
cfg = yaml.safe_load(open(yaml_file, 'r'))
T1_s, T2_s, T3_s = cfg['dataset']['T1_selected'], cfg['dataset']['T2_selected'], cfg['dataset']['T3_selected']
P_TO_CP = []
T_TO_CP = []
for p in T1_s:
    P_TO_CP += [p]
    T_TO_CP += [1]
for p in T2_s:
    P_TO_CP += [p]
    T_TO_CP += [2]
for p in T3_s:
    P_TO_CP += [p]
    T_TO_CP += [3]

data_root = r"E:\dataset\ubfc-phys\pos_rppg"
data_root_2 = r"E:\dataset\ubfc-phys\pos_rppg_3.5"
# drop_num = cfg['train']['drop_num']
drop_num = 35

def mean_filter(data, window_size):
    filtered_data = []
    for i in range(len(data)):
        if i < window_size // 2 or i >= len(data) - window_size // 2:
            filtered_data.append(data[i])
        else:
            window = data[i - window_size // 2:i + window_size // 2 + 1]
            filtered_data.append(sum(window) / window_size)
    return filtered_data


def del_nan0(check_data, reference_data, ref_x_1, ref_x_2):
    has_nan = np.isnan(check_data)
    nan_indices = np.where(has_nan)[0]
    check_data = np.delete(check_data, nan_indices)
    # reference_data = np.delete(reference_data, nan_indices)
    ref_x_1 = np.delete(ref_x_1, nan_indices)
    # ref_x_2 = np.delete(ref_x_2, nan_indices)

    has_inf = check_data == 0
    inf_indices = np.where(has_inf)[0]
    check_data = np.delete(check_data, inf_indices)
    # reference_data = np.delete(reference_data, inf_indices)
    ref_x_1 = np.delete(ref_x_1, inf_indices)
    # ref_x_2 = np.delete(ref_x_2, inf_indices)

    return check_data, reference_data, ref_x_1, ref_x_2

def compute_PR_RR(Agreement_list, input_signal_1, input_signal_2, fs_1, fs_2, show=False):
    # A wavelet-based decomposition method for a robust extraction of pulse rate from video recordings
    # 0.25s
    peak_1 = signal.find_peaks(input_signal_1, distance=14)[0]
    ppg_peaks, sig_filt, peaks_filt, thres = eng.PPG_pulses_detector(
        matlab.double(input_signal_2.tolist()),
        35.0,
        0.7,
        2.5,
        3.0,
        0.05,
        150e-3,
        1.0,
        400e-3,
        0.0,
        nargout=4)
    # try:
    #     peak_2 = np.array(ppg_peaks[0])
    # except Exception as e:
    #     print(e)
    #     print(input_signal_2)
    #     raise e
    peak_2 = np.array(ppg_peaks[0])
    PR = np.array([(x[1] - x[0]) * 1000 / fs_1 for x in zip(peak_1[:-1], peak_1[1:])])
    RR = np.array([(x[1] - x[0]) * 1000 / fs_2 for x in zip(peak_2[:-1], peak_2[1:])])
    # RR = mean_filter(RR, 5)

    x_PR = np.cumsum(PR) / 1000.0
    x_RR = np.cumsum(RR) / 1000.0

    # Recognizing Emotions Induced by Affective Sounds through Heart Rate Variability. TAC. 2015
    # 提到使用三次样条插值
    # Comparison of methods for removal of ectopy in measurement of heart rate variability
    # 插值依据

    RR, PR, x_RR, x_PR = del_nan0(RR, PR, x_RR, x_PR)
    PR = 1000 / PR
    RR = 1000 / RR

    f_PR = interp1d(x_PR, PR, kind='cubic')
    f_RR = interp1d(x_RR, RR, kind='cubic')

    # B. Mali et al., "Matlab-based tool for ECG and HRV analysis", Biomed. Signal Process. Control, vol. 10, pp. 108-116, 2014.
    # 4Hz重采样对齐
    fs = 4.0
    steps = 1 / fs

    x_PR_max = np.max(x_PR)
    x_RR_max = np.max(x_RR)

    x_PR_min = np.min(x_PR)
    x_RR_min = np.min(x_RR)

    xx = np.arange(np.max([x_PR_min, x_RR_min]), np.min([x_PR_max, x_RR_max]), steps)
    xx[0] = np.max([x_PR_min, x_RR_min])
    xx[-1] = np.min([x_PR_max, x_RR_max])

    PR = f_PR(xx)
    RR = f_RR(xx)

    mean = np.mean([PR, RR], axis=0)
    diff = PR - RR
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    hline_ceil = md + 1.96 * sd
    hline_floor = md - 1.96 * sd

    C = np.abs(PR - RR)
    C_ = np.zeros(C.shape)
    C_[C < 1.96 * sd] = 1
    C_[C >= 1.96 * sd] = 0
    Agreement = np.sum(C_) / len(C_)
    Agreement_list.append(Agreement)
    if show:
        plt.figure()
        plt.rc('axes', unicode_minus=False)
        plt.scatter(mean, diff, color='b')
        plt.axhline(md, color='black', linestyle='-')
        plt.axhline(hline_ceil, color='gray', linestyle='--')
        plt.axhline(hline_floor, color='gray', linestyle='--')
        plt.show()


Agreement_list = []
for dir_, task in zip(P_TO_CP, T_TO_CP):
    print(dir_, task)
    person_path = os.path.join(data_root, dir_)
    person_path_2 = os.path.join(data_root_2, dir_)
    # ppg

    rppg_signal = pd.read_csv(os.path.join(person_path, r"T{0}/rppg_all_{1}_T{0}.csv").format(task, dir_),
                              header=None)
    rppg_signal = rppg_signal.to_numpy().squeeze(-1)
    rppg_signal = butter_bandpass_filter(rppg_signal, 35, 0.7, 2.5)
    re_sam = math.floor(len(rppg_signal))
    # rppg_signal = signal.resample(rppg_signal, re_sam)
    # ppg_signal_all = moving_avg(ppg_signal_all, 128)
    rppg_mean = np.mean(rppg_signal)
    rppg_max = np.max(np.abs(rppg_signal))
    rppg_signal = - (rppg_signal - rppg_mean) / rppg_max

    ppg_signal = pd.read_csv(
        os.path.join(person_path_2, r"T{0}/bvp_{1}_T{0}.csv").format(task, dir_),
        header=None)
    ppg_signal = ppg_signal.to_numpy().squeeze(-1)
    # ppg_signal = detrend(ppg_signal)
    ppg_signal = butter_bandpass_filter(ppg_signal, 64, 0.7, 2.5)
    ppg_signal = signal.resample(ppg_signal, re_sam)
    # ppg_signal = moving_avg(ppg_signal, 128)
    ppg_mean = np.mean(ppg_signal)
    ppg_max = np.max(np.abs(ppg_signal))
    ppg_signal = - (ppg_signal - ppg_mean) / ppg_max

    rppg_signal = rppg_signal[drop_num:]
    ppg_signal = ppg_signal[drop_num:]

    compute_PR_RR(Agreement_list, rppg_signal, ppg_signal, 35, 35)

# Agreement Between Methods of Measurement with Multiple Observations Per Individual
# Bland-Altman图一致性
print("{:.4f}".format(np.average(Agreement_list) * 100))  # 93.7740
