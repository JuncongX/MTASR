import os
import torch
import numpy as np
import pandas as pd
import math
import cv2
from scipy import signal
from torch.utils.data import Dataset
import yaml
from utils.filter import butter_bandpass_filter, detrend
from PIL import Image
from torchvision import transforms

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


def z_score(ppg_signal):
    ppg_mean = np.mean(ppg_signal)
    ppg_signal = ppg_signal - ppg_mean
    ppg_std = np.std(ppg_signal)

    return ppg_signal / ppg_std


img_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class rPPG_Dataset(Dataset):
    def __init__(self, data_root, clip_len, jump_num, drop_num, step, person_list, task_s, label_s,
                 transform=img_transform):
        super(rPPG_Dataset, self).__init__()

        assert jump_num > 0 and type(jump_num) is int

        self.clip_len = clip_len  # 切片长度
        self.img_paths, self.labels, self.tasks, self.PPG_ = [], [], [], []  # 图片路径与标签
        self.jump_num = jump_num  # 跳几帧
        self.drop_num = drop_num  # 去掉视频之前的几帧
        self.step = step  # 滑动窗口步长
        self.transform = transform

        for _, (dir_, task, label) in enumerate(zip(person_list, task_s, label_s)):
            person_path = os.path.join(data_root, dir_)
            # df_info = pd.read_csv(os.path.join(person_path, r'info_{0}.txt'.format(dir_)), header=None)
            # label = 1 if df_info.values[2][0] == "test" else 0  # 1为测试组 0为对照组

            # # AU_pose_gaze
            # more_info_estimation = pd.read_csv(os.path.join(person_path, r"T{0}/vid_{1}_T{0}.csv").format(task, dir_))

            # image
            imgs_root_path = os.path.join(person_path, r"T{0}".format(task), "resized_face")
            imgs_names = os.listdir(imgs_root_path)
            imgs_names.sort(key=lambda x: int(x.split('.')[0]))

            # ppg
            ppg_signal = pd.read_csv(os.path.join(person_path, r"T{0}/bvp_p_{1}_T{0}.csv").format(task, dir_),
                                     header=None)
            ppg_signal = ppg_signal.to_numpy().squeeze(-1)
            # ppg_signal = detrend(ppg_signal)
            # ppg_signal = butter_bandpass_filter(ppg_signal, 64, 0.7, 2.5)
            ppg_signal = signal.resample(ppg_signal, len(imgs_names))  # 对ppg信号重采样

            imgs_names = imgs_names[self.drop_num:]
            ppg_signal = ppg_signal[self.drop_num:]

            for times_index in range(self.jump_num):
                total_times = (len(ppg_signal) - self.clip_len * self.jump_num) / self.step + 1
                for times in range(math.floor(total_times)):
                    self.labels.append(label)
                    self.tasks.append(task)
                    # ppg_start_index = img_index + times_index - (self.clip_len - 1) * self.jump_num
                    ppg_start_index = int(times_index + times * self.step)
                    ppg_end_index = int(ppg_start_index + self.clip_len * self.jump_num)
                    self.PPG_.append(z_score(ppg_signal[ppg_start_index: ppg_end_index: self.jump_num]))

                    img_clip = imgs_names[ppg_start_index: ppg_end_index: self.jump_num]
                    for i in range(len(img_clip)):
                        img_clip[i] = os.path.join(imgs_root_path, img_clip[i])
                    self.img_paths.append(img_clip)

        self.PPG = torch.from_numpy(np.array(self.PPG_))
        self.Labels = torch.from_numpy(np.array(self.labels).astype(np.uint8))
        self.Tasks = torch.from_numpy(np.array(self.tasks).astype(np.uint8))

    def load_frames(self, clip_array):
        frames = []

        for i, frame_name in enumerate(clip_array):
            image = Image.open(frame_name)
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        return torch.stack(frames).permute(1, 0, 2, 3)

    # def normalize(self, buffer):
    #     for i, frame in enumerate(buffer):
    #         frame -= np.array([[[90.0, 98.0, 102.0]]])
    #         buffer[i] = frame
    #
    #     return buffer

    # def to_tensor(self, buffer):
    #     return buffer.transpose((3, 0, 1, 2))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        buffer = self.load_frames(self.img_paths[item])

        return self.Labels[item].to(torch.long), \
               self.Tasks[item].to(torch.long), \
               self.PPG[item].unsqueeze(dim=0).to(torch.float32), \
               buffer


if __name__ == '__main__':
    import yaml
    from torch.utils.data import DataLoader

    yaml_file = "./setting.yaml"
    cfg = yaml.safe_load(open(yaml_file, 'r'))
    # data_root, clip_len, jump_num, drop_num, step
    person_list, tasks, labels = data_selected()
    dataset = rPPG_Dataset(r"/home/ps/outmounts/data2/xjc/ubfc_phys/train_rppg/", cfg['train']['clip_len'],
                           cfg['train']['jump_num'], cfg['train']['drop_num'], cfg['train']['step'], person_list, tasks,
                           labels)

    loader = DataLoader(dataset=dataset, batch_size=4, num_workers=2)
    for i, data in enumerate(loader):
        labels, tasks, bvp, imgs = data
        print(bvp.shape)
        print(imgs.shape)
        if i == 2:
            break
        # print(inputs.shape, labels, tasks)
