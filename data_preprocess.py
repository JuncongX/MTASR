from scipy import signal
from scipy import interpolate
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import math
import os
import yaml
import csv

from utils.HRV import HRV
from utils.filter import butter_bandpass_filter, detrend

yaml_file = "./setting.yaml"
cfg = yaml.safe_load(open(yaml_file, 'r'))
data_root = r"/home/som/8T/DataSets/ubfc_phys/pos_rppg/"
data_root_2 = r"/home/som/8T/DataSets/ubfc_phys/train/"
drop_num = cfg['train']['drop_num']
step = cfg['train']['step']
clip_len = cfg['train']['clip_len']


def outlier(ppg_signal):
    ppg_signal_new = ppg_signal
    ppg_mean = np.mean(ppg_signal)
    ppg_std = np.std(ppg_signal)
    outlier = np.logical_or(ppg_signal > ppg_mean + 2 * ppg_std, ppg_signal < ppg_mean - 2 * ppg_std)
    index = np.arange(len(ppg_signal))
    tck = interpolate.splprep(index[~outlier], ppg_signal[~outlier])
    for x in index[outlier]:
        ppg_signal_new[x] = interpolate.splev(x, tck)
    return ppg_signal_new


def missing_fix(signal):
    signal_new = signal
    missing = np.zeros(signal.shape, dtype=np.bool8)
    missing[signal == 0] = True
    if np.sum(missing) == 0:
        # return move_signal_filter(signal)
        return signal
    missing[signal != 0] = False
    index = np.arange(len(signal))
    try:
        tck = interpolate.splrep(index[~missing], signal[~missing])
        for x in index[missing]:
            signal_new[x] = interpolate.splev(x, tck)
    except Exception as e:
        print(e)
        # return move_signal_filter(signal)
        return signal
    # return move_signal_filter(signal_new)
    return signal_new


def move_signal_filter(x):
    return savgol_filter(x, 11, 3)


def find_peak(ppg_signal):
    # ppg_signal = signal.resample(ppg_signal, math.floor(len(ppg_signal) / 2))
    peak_ = signal.find_peaks(ppg_signal, distance=10)[
        0]  # 300ms Reference: Robust PPG Peak Detection Using Dilated Convolutional Neural Networks
    index_arr = np.zeros((len(ppg_signal)), dtype="uint8")
    for index in peak_:
        index_arr[index] = 1
    return index_arr


def z_score(ppg_signal):
    ppg_mean = np.mean(ppg_signal)
    ppg_signal = ppg_signal - ppg_mean
    ppg_std = np.std(ppg_signal)

    return ppg_signal / ppg_std


# def norm(ppg_signal):
#     ppg_mean = np.mean(ppg_signal)
#     ppg_max = np.max(ppg_signal)
#     ppg_min = np.min(ppg_signal)
#     ppg_signal = (ppg_signal - ppg_mean) / (ppg_max - ppg_min)
#     return ppg_signal


def get_data(task, dir_, save_datas):
    person_path = os.path.join(data_root, dir_)
    person_path_2 = os.path.join(data_root_2, dir_)

    ppg_signal = pd.read_csv(os.path.join(person_path, r"T{0}/rppg_all_{1}_T{0}.csv").format(task, dir_),
                             header=None)
    ppg_signal = ppg_signal.to_numpy().squeeze(-1)

    bvp_signal = pd.read_csv(os.path.join(person_path_2, r"T{0}/bvp_{1}_T{0}.csv").format(task, dir_),
                             header=None)
    bvp_signal = bvp_signal.to_numpy().squeeze(-1)
    bvp_signal = signal.resample(bvp_signal, len(ppg_signal))

    ppg_signal = ppg_signal[drop_num:]
    bvp_signal = bvp_signal[drop_num:]
    bvp_signal = z_score(bvp_signal)
    bvp_signal = butter_bandpass_filter(bvp_signal, 35, 0.7, 2.5)

    df_info = pd.read_csv(os.path.join(person_path_2, r'info_{0}.txt'.format(dir_)), header=None)
    if task == 1:
        level = 2
    else:
        if df_info.values[2][0] == "test":
            level = 1
        else:
            level = 0

    # gaze_0_x gaze_0_y gaze_0_z gaze_1_x gaze_1_y gaze_1_z
    # pose_Rx pose_Ry pose_Rz
    movement_info = pd.read_csv(os.path.join(person_path_2, r"T{0}/vid_{1}_T{0}.csv").format(task, dir_),
                                usecols=[
                                    5, 6, 7, 8, 9, 10, 296, 297, 298
                                ])
    movement_info = movement_info.to_numpy()
    movement_info = movement_info[drop_num:]

    total_times = (len(ppg_signal) - clip_len) / step + 1

    label = 1 if task == 2 or task == 3 else 0
    for times in range(math.floor(total_times)):
        # ppg_start_index = img_index + times_index - (self.clip_len - 1) * self.jump_num
        ppg_start_index = int(times * step)
        ppg_end_index = int(ppg_start_index + clip_len)
        ppg_signal_ = ppg_signal[ppg_start_index: ppg_end_index]
        vpg_signal_ = np.gradient(ppg_signal_)
        bvp_signal_ = bvp_signal[ppg_start_index: ppg_end_index]
        bvp_signal_ = z_score(bvp_signal_)
        bvp_hrv_ = HRV(bvp_signal_, 35, 14)  # 35Hz/2.5Hz = 14
        HR = bvp_hrv_.get_hr()

        # ppg_signal_ = outlier(ppg_signal_)
        # ppg_signal_ = butter_bandpass_filter(ppg_signal_, 35)
        # ppg_signal_ = detrend(ppg_signal_)
        # ppg_signal_ = norm(ppg_signal_)
        ppg_signal_ = z_score(ppg_signal_)
        vpg_signal_ = z_score(vpg_signal_)

        peak_ = find_peak(ppg_signal_)
        vpg_peak_ = find_peak(vpg_signal_)
        movement_info_ = movement_info[ppg_start_index: ppg_end_index]
        # print("cwtmatr_", cwtmatr_.shape)
        # print("movement_info_", movement_info_.shape)
        datas = [dir_, task, level, label, ppg_signal_, peak_, vpg_signal_, vpg_peak_, HR]

        # mi_list = []
        # mistd_list = []
        # for i in range(movement_info_.shape[1]):
        #     mi = missing_fix(movement_info_[:, i])
        #     mi_list.append(np.gradient(mi))
        #     mistd_list.append(np.std(mi, ddof=1))
        # datas = datas + mi_list + mistd_list
        datas = datas + [np.gradient(z_score(missing_fix(movement_info_[:, i]))) for i in
                         range(movement_info_.shape[1])]
        save_datas.append(datas)


if __name__ == '__main__':
    havent_done = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's13', 's14', 's15',
                   's16', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
                   's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
                   's44', 's45', 's46', 's48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']
    save_datas = []
    for dir_ in havent_done:
        person_path = os.path.join(data_root, dir_)
        for task in [1, 2, 3]:
            print(dir_, task)
            get_data(task, dir_, save_datas)
    np.save("ubfc_phys_new.npy", save_datas)
