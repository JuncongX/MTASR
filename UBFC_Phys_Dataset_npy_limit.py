import os
import torch
import numpy as np
import pandas as pd
import math
from scipy import signal
from torch.utils.data import Dataset
import yaml

yaml_file = "./setting.yaml"
cfg = yaml.safe_load(open(yaml_file, 'r'))
T1_s, T2_s, T3_s = cfg['dataset']['T1_selected'], cfg['dataset']['T2_selected'], cfg['dataset']['T3_selected']
T1_selected, T2_selected, T3_selected = set(T1_s), set(T2_s), set(T3_s)


def data_selected():
    stress_2 = T2_selected - (T2_selected & T3_selected)

    person_list, tasks, labels = [], [], []
    person_list += T1_s
    tasks += [1] * len(T1_s)
    person_list += T3_s
    tasks += [3] * len(T3_s)

    for s_2 in stress_2:
        person_list.append(s_2)
        tasks.append(2)

    labels += [0] * len(T1_s)
    labels += [1] * (len(T3_s) + len(stress_2))

    return person_list, tasks, labels


def data_selected_level(data_root):
    person_list, tasks, level = [], [], []
    # person_list += T1_s
    # tasks += [1] * len(T1_s)
    person_list += T2_s
    tasks += [2] * len(T2_s)
    person_list += T3_s
    tasks += [3] * len(T3_s)

    for person, task in zip(person_list, tasks):
        df_info = pd.read_csv(os.path.join(data_root, person, r'info_{0}.txt'.format(person)), header=None)
        if df_info.values[2][0] == "test":
            level += [1]
        else:
            level += [0]
    return person_list, tasks, level


class rPPG_Dataset(Dataset):
    def __init__(self, person_list, task_s, label_s):
        super(rPPG_Dataset, self).__init__()

        data = np.load(r"ubfc_phys_new.npy", allow_pickle=True)
        needed_data = None
        for _, (person, task, label) in enumerate(zip(person_list, task_s, label_s)):
            selected_data = data[(data[:, 0] == person) & (data[:, 1] == task) & (data[:, 3] == label)]
            if needed_data is None:
                needed_data = selected_data
            else:
                needed_data = np.concatenate((needed_data, selected_data))

        self.labels = torch.from_numpy(needed_data[:, 3].astype(np.uint8))
        self.tasks = torch.from_numpy(needed_data[:, 1].astype(np.uint8))
        self.level = torch.from_numpy(needed_data[:, 2].astype(np.uint8))
        self.PPG = torch.Tensor([ppg.astype(np.float32) for ppg in needed_data[:, 4]])
        self.peak = torch.Tensor([peak_.astype(np.uint8) for peak_ in needed_data[:, 5]])
        self.VPG = torch.Tensor([ppg.astype(np.float32) for ppg in needed_data[:, 6]])
        self.vpg_peak = torch.Tensor([peak_.astype(np.uint8) for peak_ in needed_data[:, 7]])
        self.HR = torch.Tensor(needed_data[:, 8].astype(np.float32))
        self.gaze = torch.Tensor(
            [[gaze_[i].astype(np.float32) for i in range(6)] for gaze_ in needed_data[:, 9:15]])
        self.pose = torch.Tensor(
            [[pose_[i].astype(np.float32) for i in range(3)] for pose_ in needed_data[:, 15:18]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.labels[item].to(torch.long), \
               self.tasks[item].to(torch.long), \
               self.level[item].to(torch.long), \
               self.PPG[item].unsqueeze(dim=0).to(torch.float), \
               self.peak[item].to(torch.float), \
               self.VPG[item].unsqueeze(dim=0).to(torch.float), \
               self.vpg_peak[item].to(torch.float), \
               self.HR[item].to(torch.float), \
               self.gaze[item].to(torch.float), \
               self.pose[item].to(torch.float)


class rPPG_Dataset_level(Dataset):
    def __init__(self, person_list, task_s, level_s):
        super(rPPG_Dataset_level, self).__init__()

        data = np.load(r"ubfc_phys_new.npy", allow_pickle=True)
        needed_data = None
        for _, (person, task, level_) in enumerate(zip(person_list, task_s, level_s)):
            selected_data = data[(data[:, 0] == person) & (data[:, 1] == task) & (data[:, 2] == level_)]
            if needed_data is None:
                needed_data = selected_data
            else:
                needed_data = np.concatenate((needed_data, selected_data))

        self.labels = torch.from_numpy(needed_data[:, 3].astype(np.uint8))
        self.tasks = torch.from_numpy(needed_data[:, 1].astype(np.uint8))
        self.level = torch.from_numpy(needed_data[:, 2].astype(np.uint8))
        self.PPG = torch.Tensor([ppg.astype(np.float32) for ppg in needed_data[:, 4]])
        self.peak = torch.Tensor([peak_.astype(np.uint8) for peak_ in needed_data[:, 5]])
        self.VPG = torch.Tensor([ppg.astype(np.float32) for ppg in needed_data[:, 6]])
        self.vpg_peak = torch.Tensor([peak_.astype(np.uint8) for peak_ in needed_data[:, 7]])
        self.HR = torch.Tensor(needed_data[:, 8].astype(np.float32))
        self.gaze = torch.Tensor(
            [[gaze_[i].astype(np.float32) for i in range(6)] for gaze_ in needed_data[:, 9:15]])
        self.pose = torch.Tensor(
            [[pose_[i].astype(np.float32) for i in range(3)] for pose_ in needed_data[:, 15:18]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.labels[item].to(torch.long), \
               self.tasks[item].to(torch.long), \
               self.level[item].to(torch.long), \
               self.PPG[item].unsqueeze(dim=0).to(torch.float), \
               self.peak[item].to(torch.float), \
               self.VPG[item].unsqueeze(dim=0).to(torch.float), \
               self.vpg_peak[item].to(torch.float), \
               self.HR[item].to(torch.float), \
               self.gaze[item].to(torch.float), \
               self.pose[item].to(torch.float)


if __name__ == '__main__':
    import yaml
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    from sklearn.model_selection import StratifiedKFold
    splits = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    person_list, tasks, labels = data_selected()
    person_list, tasks, labels = np.array(person_list), np.array(tasks), np.array(labels)
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(person_list)), labels)):
        train_p, train_t, train_l = person_list[train_idx], tasks[train_idx], labels[train_idx]
        val_p, val_t, val_l = person_list[val_idx], tasks[val_idx], labels[val_idx]
        print(val_l)
    dataset = rPPG_Dataset_level(person_list, tasks, labels)

    data_selected_level(r"/home/som/8T/DataSets/ubfc_phys/train/")
    loader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)
    for i, data in enumerate(loader):
        labels, tasks, level, bvp, peak, vpg, vpg_peak, HR, gaze, pose = data
        print(level)
        print(tasks, type(level), labels)
        print(gaze)
        print(gaze.shape)
        print(pose.shape)
        # print(AU.shape)

        bvp = bvp.squeeze(0).squeeze(0)
        # bvp = signal.resample(bvp, int(len(bvp) / 2))

        print(bvp.shape)
        index = 0
        peak_count = 0
        for p in peak[0]:
            if p:
                plt.plot(index, bvp[index], marker='o', color='r')
                peak_count += 1
            index += 1
        print(peak_count)
        plt.plot(range(bvp.shape[0]), bvp, label='bvp')
        plt.legend()
        plt.show()
