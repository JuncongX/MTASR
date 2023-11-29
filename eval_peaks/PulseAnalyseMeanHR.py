import matlab
import matlab.engine
import numpy as np
import yaml
import math
import os
from scipy import signal
import pandas as pd
from utils.filter import butter_bandpass_filter, moving_avg, detrend
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

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

data_root = r"E:\dataset\ubfc-phys\green_rppg"
data_root_2 = r"E:\dataset\ubfc-phys\pos_rppg_3.5"
# drop_num = cfg['train']['drop_num']
drop_num = 35

eng = matlab.engine.start_matlab()

clip_len = 1050
step = 35

# 0.2 300e-3
# POS: 4.722712059276187  0.4638889994144781  7.5345886412800525
# CHROM: 4.724146285331937  0.4529616506267179  7.555464908331538
# 2SR: 6.658540059295478  0.2651052577191502  9.547525092270774
# ICA: 5.670801053148075  0.40706500148336644  8.38251604802973
# Green: 6.454475004429801  0.3245189049525386  9.139886959627225

# 0.05 400e-3
# POS: 2.6228468848248623  0.6780829831144183  3.4126839830908446
# CHROM: 2.811691567817382  0.6746900343623313  3.658684480507919
# 2SR: 6.369751353553454  0.36850534950702896  7.821695531443303
# ICA: 3.9618415432077354  0.5812287194980739  4.920110974128712
# Green: 4.797608397863598  0.46432856464881217  5.959677715706581


def get_labels(val_p, val_t):
    label = []
    for person, task in zip(val_p, val_t):
        if task == 1:
            label += [0]
        else:
            label += [1]
    return label


def normalize(arr):
    return 2 * ((arr - arr.min()) / (arr.max() - arr.min())) - 1


def med_filter(rri, L=51):
    RRi = rri
    PP1_flip = np.flip(RRi[2: int((L - 1) / 2 + 1)])
    PPlast_flip = np.flip(RRi[int(-((L - 1) / 2 + 1)):-2])
    RRi_flip = np.concatenate((PP1_flip, RRi, PPlast_flip))
    for i in range(len(RRi_flip) - L + 1):
        med_win = RRi_flip[i: i + L]
        medRR = np.median(med_win)
        if np.abs(RRi[i] - medRR) >= 150:
            RRi[i] = medRR
    return RRi


def analysis(y_test, y_predict):
    mse = np.sum((y_test - y_predict) ** 2) / len(y_test)
    rmse = np.sqrt(mse)
    mae = np.sum(np.absolute(y_test - y_predict)) / len(y_test)
    r2 = 1 - mse / np.var(y_test)
    return mse, rmse, mae, r2


MAE_list = []
pearson_list = []
RMSE_list = []
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
    rppg_signal = signal.resample(rppg_signal, re_sam)
    # ppg_signal_all = moving_avg(ppg_signal_all, 128)
    rppg_signal = normalize(rppg_signal)

    ppg_signal = pd.read_csv(
        os.path.join(person_path_2, r"T{0}/bvp_{1}_T{0}.csv").format(task, dir_),
        header=None)
    ppg_signal = ppg_signal.to_numpy().squeeze(-1)
    # ppg_signal = detrend(ppg_signal)
    ppg_signal = butter_bandpass_filter(ppg_signal, 64, 0.7, 2.5)
    ppg_signal = signal.resample(ppg_signal, re_sam)
    # ppg_signal = moving_avg(ppg_signal, 128)
    ppg_signal = normalize(ppg_signal)

    rppg_signal = rppg_signal[drop_num:]
    ppg_signal = ppg_signal[drop_num:]

    # print(';'.join(map(str, normalize(ppg_signal[350:1400]))))
    IHR, IPR = [], []
    for i in range(math.floor((len(rppg_signal) - clip_len) / step) + 1):
        ppg_peaks, sig_filt, peaks_filt, thres = eng.PPG_pulses_detector(
            matlab.double(normalize(ppg_signal[step * i:step * i + clip_len]).tolist()),
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
        try:
            ppg_peaks = np.array(ppg_peaks[0])
        except Exception as e:
            print(e)
            print(ppg_peaks)
            continue

        rppg_peaks = signal.find_peaks(rppg_signal[step * i:step * i + clip_len], distance=14)[0]

        RR = np.array([(x[1] - x[0]) * 1000 / 35.0 for x in zip(ppg_peaks[:-1], ppg_peaks[1:])])
        PP = np.array([(x[1] - x[0]) * 1000 / 35.0 for x in zip(rppg_peaks[:-1], rppg_peaks[1:])])

        RR = med_filter(RR)
        PP = med_filter(PP)

        HR = 60000 / np.average(RR)
        PR = 60000 / np.average(PP)

        IHR.append(HR)
        IPR.append(PR)
    IHR = np.array(IHR)
    IPR = np.array(IPR)


    def del_infnan(check_data, reference_data):
        has_nan = np.isnan(check_data)
        nan_indices = np.where(has_nan)[0]
        check_data = np.delete(check_data, nan_indices)
        reference_data = np.delete(reference_data, nan_indices)

        has_inf = np.isinf(check_data)
        inf_indices = np.where(has_inf)[0]
        check_data = np.delete(check_data, inf_indices)
        reference_data = np.delete(reference_data, inf_indices)

        return check_data, reference_data


    IHR, IPR = del_infnan(IHR, IPR)
    IPR, IHR = del_infnan(IPR, IHR)

    mse_a, rmse_a, mae_a, r2_a = analysis(IHR, IPR)
    try:
        p_a = pearsonr(IHR, IPR)[0]
    except Exception as e:
        print(e)
        print(IHR)
        print(IPR)
        continue
    # try:
    #     p_a = pearsonr(IHR, IPR)[0]
    # except Exception as e:
    #     print(e)
    #     continue
    print(mae_a)
    print(p_a)
    print("==============")
    MAE_list.append(mae_a)
    pearson_list.append(p_a)
    RMSE_list.append(rmse_a)
print("TOTAL AVG: {0}  {1}  {2}".format(np.mean(MAE_list), np.mean(pearson_list), np.mean(RMSE_list)))
# print("TOTAL STD: {0}  {1}".format(np.std(MAE_list), np.std(pearson_list)))
