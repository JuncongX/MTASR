# reference: Kotzen K, Charlton P H, Landesberg A, et al. Benchmarking photoplethysmography peak detection algorithms using the electrocardiogram signal as a reference[C]//2021 Computing in Cardiology (CinC). IEEE, 2021, 48: 1-4.
# reference: Han D, Bashar S K, Lázaro J, et al. A real-time ppg peak detection method for accurate determination of heart rate during sinus rhythm and cardiac arrhythmia[J]. Biosensors, 2022, 12(2): 82.
# reference: Lin W H, Zheng D, Li G, et al. Investigation on pulse wave forward peak detection and its applications in cardiovascular health[J]. IEEE Transactions on Biomedical Engineering, 2021, 69(2): 700-709.
# reference: Sabour R M, Benezeth Y, De Oliveira P, et al. Ubfc-phys: A multimodal database for psychophysiological studies of social stress[J]. IEEE Transactions on Affective Computing, 2021.
# reference: https://github.com/marianux/ecg-kit.git
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
clip_len = 1050
step = 35

eng = matlab.engine.start_matlab()


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


total_count = 0
correct_count = 0
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


        RR, PP = del_infnan(RR, PP)
        PP, RR = del_infnan(PP, RR)

        HR = 60000 / np.average(RR)
        PR = 60000 / np.average(PP)

        total_count += 1
        if PR >= HR - 5 and PR <= HR + 5:
            correct_count += 1
print(correct_count / total_count * 100)  # 84.65986394557822
