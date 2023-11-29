import numpy as np
import os
import yaml

yaml_file = "./setting.yaml"
cfg = yaml.safe_load(open(yaml_file, 'r'))

device_list = cfg['train']['device_list']
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
from models.MTASR import MetaStress
from UBFC_Phys_Dataset_npy_limit import rPPG_Dataset, data_selected
import random

seed = cfg['train']['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
LR = cfg['train']['LR']
weight_pp = cfg['train']['weight_pp']
weight_hr = cfg['train']['weight_hr']


class P_HR_C_loss(nn.Module):
    def __init__(self):
        super(P_HR_C_loss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.hr_loss = nn.L1Loss()

    def forward(self, classify_result, classify_labels, peak_result, peak_labels, hr_result, hr_gt):
        cl_loss = self.classify_loss(classify_result, classify_labels)
        pp_loss = F.binary_cross_entropy(peak_result, peak_labels)
        hr_loss = self.hr_loss(hr_result, hr_gt)
        return cl_loss + weight_pp * pp_loss + weight_hr * hr_loss


class P_HR_loss(nn.Module):
    def __init__(self):
        super(P_HR_loss, self).__init__()
        self.hr_loss = nn.L1Loss()

    def forward(self, peak_result, peak_labels, hr_result, hr_gt):
        return F.binary_cross_entropy(peak_result, peak_labels) + self.hr_loss(hr_result, hr_gt)


def train_epoch(net, device, data_loader, criterion_PHC, criterion_PH, optimizer, net_type):
    net.train()
    train_loss, train_correct, MAE = 0.0, 0.0, 0.0
    for i, data in enumerate(data_loader):
        labels, tasks, level, bvp, peak, vpg, vpg_peak, HR, gaze, pose = data
        labels, tasks, bvp, peak, HR = Variable(labels).to(device), Variable(tasks).to(device), Variable(bvp).to(
            device), Variable(peak).to(device), Variable(HR).to(device)

        optimizer.zero_grad()
        if net_type == "hr":
            hr, p_peak = net(bvp, net_type)
            loss = criterion_PH(p_peak, peak, hr.squeeze(-1), HR)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bvp.size(0)
            MAE += (hr.squeeze(-1) - HR).abs().mean()

        elif net_type == "both":
            outputs, hr, p_peak = net(bvp, net_type)
            loss = criterion_PHC(outputs, labels, p_peak, peak, hr.squeeze(-1), HR)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            train_loss += loss.item() * bvp.size(0)
            scores, predictions = torch.max(outputs.data, 1)
            train_correct += (predictions == labels).sum().item()
            MAE += (hr.squeeze(-1) - HR).abs().mean()
    if net_type == "hr":
        return train_loss, MAE / len(data_loader)
    elif net_type == "both":
        return train_loss, train_correct, MAE / len(data_loader)


def valid_epoch(net, device, data_loader, criterion_PHC, criterion_PH, net_type):
    net.eval()
    valid_loss, val_correct, MAE, TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i, data in enumerate(data_loader):
        labels, tasks, level, bvp, peak, vpg, vpg_peak, HR, gaze, pose = data
        labels, tasks, bvp, peak, HR = Variable(labels).to(device), Variable(tasks).to(device), Variable(bvp).to(
            device), Variable(peak).to(device), Variable(HR).to(device)
        if net_type == "hr":
            hr, p_peak = net(bvp, net_type)
            loss = criterion_PH(p_peak, peak, hr.squeeze(-1), HR)
            valid_loss += loss.item() * bvp.size(0)

            MAE += (hr.squeeze(-1) - HR).abs().mean()
        elif net_type == "both":
            output_test, hr, p_peak = net(bvp, net_type)
            loss = criterion_PHC(output_test, labels, p_peak, peak, hr.squeeze(-1), HR)
            _, predicted = torch.max(output_test, 1)
            valid_loss += loss.item() * bvp.size(0)
            scores, predictions = torch.max(output_test.data, 1)
            val_correct += (predictions == labels).sum().item()
            MAE += (hr.squeeze(-1) - HR).abs().mean()

            TP += np.sum((labels.detach().cpu().numpy() == 1) & (predicted.detach().cpu().numpy() == 1))
            TN += np.sum((labels.detach().cpu().numpy() == 0) & (predicted.detach().cpu().numpy() == 0))
            FP += np.sum((labels.detach().cpu().numpy() == 0) & (predicted.detach().cpu().numpy() == 1))
            FN += np.sum((labels.detach().cpu().numpy() == 1) & (predicted.detach().cpu().numpy() == 0))
    if net_type == "hr":
        return valid_loss, MAE / len(data_loader)
    elif net_type == "both":
        return valid_loss, val_correct, MAE / len(data_loader), TP, TN, FP, FN


hr_times = 50
epoch_whole = cfg['train']['epoch'] + hr_times
if __name__ == '__main__':
    k = cfg['train']['cv_times']
    splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    foldperf = {}
    person_list, tasks, labels = data_selected()
    person_list, tasks, labels = np.array(person_list), np.array(tasks), np.array(labels)
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(person_list)), labels)):
        print('Fold {}'.format(fold + 1))

        batch_size = cfg['train']['batch_size']
        num_workers = cfg['train']['num_workers']

        train_p, train_t, train_l = person_list[train_idx], tasks[train_idx], labels[train_idx]
        val_p, val_t, val_l = person_list[val_idx], tasks[val_idx], labels[val_idx]

        print('train subject:')
        print(list(train_p))
        print('train task:')
        print(list(train_t))
        print('test subject')
        print(list(val_p))
        print('test task:')
        print(list(val_t))

        train_dataset = rPPG_Dataset(train_p, train_t, train_l)
        test_dataset = rPPG_Dataset(val_p, val_t, val_l)

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        net = MetaStress().to(device)
        net = torch.nn.DataParallel(net)

        # Loss
        # criterion = nn.CrossEntropyLoss()
        # criterion = FocalLoss(gamma=2, alpha=0.9)
        criterion_PHC, criterion_PH = P_HR_C_loss(), P_HR_loss()

        optimizer = optim.Adam(
            net.parameters(),
            lr=LR,
        )
        # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'train_mae': [], 'test_mae': [],
                   'precision': [], 'recall': [], 'F1_score': []}

        best_perform = 0.0
        best_perform_mae = np.inf
        for epoch in range(1, epoch_whole + 1):
            if epoch <= hr_times:
                train_loss, train_MAE = train_epoch(net, device, train_loader, criterion_PHC, criterion_PH, optimizer,
                                                    "hr")
                test_loss, test_MAE = valid_epoch(net, device, test_loader, criterion_PHC, criterion_PH, "hr")
                print(
                    "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training MAE {:.2f} AVG Test MAE {:.2f} ".format(
                        epoch,
                        epoch_whole,
                        train_loss / len(train_loader.sampler),
                        test_loss / len(test_loader.sampler),
                        train_MAE,
                        test_MAE
                    ))
            else:
                train_loss, train_correct, train_MAE = train_epoch(net, device, train_loader, criterion_PHC,
                                                                   criterion_PH,
                                                                   optimizer, "both")
                test_loss, test_correct, test_MAE, TP, TN, FP, FN = valid_epoch(net, device, test_loader, criterion_PHC,
                                                                                criterion_PH, "both")

                train_loss = train_loss / len(train_loader.sampler)
                train_acc = train_correct / len(train_loader.sampler) * 100
                test_loss = test_loss / len(test_loader.sampler)
                test_acc = test_correct / len(test_loader.sampler) * 100

                test_precision = TP / (TP + FP) * 100
                test_recall = TP / (TP + FN) * 100
                test_F1_score = 2 * test_precision * test_recall / (test_precision + test_recall)

                train_MAE, test_MAE = train_MAE.detach().cpu().numpy(), test_MAE.detach().cpu().numpy()

                if cfg['train'][
                    'save_model'] and test_acc > best_perform:
                    history['train_loss'].append(train_loss)
                    history['test_loss'].append(test_loss)
                    history['train_acc'].append(train_acc)
                    history['test_acc'].append(test_acc)
                    history['precision'].append(test_precision)
                    history['recall'].append(test_recall)
                    history['F1_score'].append(test_F1_score)
                    history['train_mae'].append(train_MAE)
                    history['test_mae'].append(test_MAE)
                    best_perform = test_acc
                    best_perform_mae = test_MAE
                    torch.save(net.module,
                               os.path.join(cfg['train']['save_path'],
                                            'MTASR_pa{2}_hr{3}_{0}_{1}.pth'.format(fold + 1,
                                                                                              cfg['train']['clip_len'],
                                                                                              weight_pp, weight_hr)))
                elif cfg['train'][
                    'save_model'] and test_acc == best_perform and best_perform_mae >= test_MAE:
                    history['train_loss'].append(train_loss)
                    history['test_loss'].append(test_loss)
                    history['train_acc'].append(train_acc)
                    history['test_acc'].append(test_acc)
                    history['precision'].append(test_precision)
                    history['recall'].append(test_recall)
                    history['F1_score'].append(test_F1_score)
                    history['train_mae'].append(train_MAE)
                    history['test_mae'].append(test_MAE)
                    best_perform_mae = test_MAE
                    torch.save(net.module,
                               os.path.join(cfg['train']['save_path'],
                                            'MTASR_pa{2}_hr{3}_{0}_{1}.pth'.format(fold + 1,
                                                                                              cfg['train']['clip_len'],
                                                                                              weight_pp, weight_hr)))

                print(
                    "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} % AVG Training MAE {:.2f} AVG Test MAE {:.2f}  precision {:.2f} % recall {:.2f} % F1_score {:.2f} %".format(
                        epoch,
                        epoch_whole,
                        train_loss,
                        test_loss,
                        train_acc,
                        test_acc,
                        train_MAE,
                        test_MAE,
                        test_precision,
                        test_recall,
                        test_F1_score
                    ))

        foldperf['fold{}'.format(fold + 1)] = history

    testl_f, tl_f, testa_f, ta_f, te_precision, te_recall, te_F1_score, ta_mae, te_mae = [], [], [], [], [], [], [], [], []
    for f in range(1, k + 1):
        testa = foldperf['fold{}'.format(f)]['test_acc']
        testa_max = np.max(testa)
        r = np.where(testa == testa_max)
        testa_f.append(testa_max)
        ta_f.append(foldperf['fold{}'.format(f)]['train_acc'][r[0][0]])
        testl_f.append(foldperf['fold{}'.format(f)]['test_loss'][r[0][0]])
        tl_f.append(foldperf['fold{}'.format(f)]['train_loss'][r[0][0]])
        te_precision.append(foldperf['fold{}'.format(f)]['precision'][r[0][0]])
        te_recall.append(foldperf['fold{}'.format(f)]['recall'][r[0][0]])
        te_F1_score.append(foldperf['fold{}'.format(f)]['F1_score'][r[0][0]])
        ta_mae.append(foldperf['fold{}'.format(f)]['train_mae'][r[0][0]])
        te_mae.append(foldperf['fold{}'.format(f)]['test_mae'][r[0][0]])

    print('Performance of {} fold cross validation'.format(k))
    print(
        "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f} \t Average Training MAE: {:.2f} \t Average Test MAE: {:.2f} \t Average Test precision: {:.2f} \t Average Test recall: {:.2f} \t Average Test F1_score: {:.2f}".format(
            np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f), np.mean(ta_mae), np.mean(te_mae),
            np.mean(te_precision), np.mean(te_recall), np.mean(te_F1_score)))
    #
    # if cfg['train']['save_model']:
    #     torch.save(net.module,
    #                os.path.join(cfg['train']['save_path'], 'meta_stress.pth'))
