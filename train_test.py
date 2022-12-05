import torch
import torch.nn as nn
from torch.utils import data
import os
import math
import numpy as np
from tqdm.auto import tqdm

from dataset import vipl_hr
from evaluate import metric, postprocess
from configs import running
from model import loss_function, physformer


def cross_validation(folds, path, train_config, test_config, methods=None):
    """
    受试者独立交叉验证
    :param folds:
    :param path: for saving models
    :param train_config:
    :param test_config:
    :param methods:
    :return:
    """
    if methods is None:
        methods = ["MAE", "RMSE", "MAPE", "R"]
    result = metric.Accumulate(len(methods))
    for i in range(folds):
        print(f"================Fold{i + 1}================")
        train_config.folds = [j for j in range(1, 6) if j != i + 1]
        test_config.folds = [i + 1]
        train_set = vipl_hr.VIPL_HR(train_config)
        test_set = vipl_hr.VIPL_HR(test_config)
        train_iter = data.DataLoader(train_set, batch_size=train_config.batch_size,
                                     shuffle=True)
        test_iter = data.DataLoader(test_set, batch_size=test_config.batch_size,
                                    shuffle=False)
        # init and train
        # net = net.to(train_config.device)
        net = nn.DataParallel(physformer.ViT_ST_ST_Compact3_TDC_gra_sharp(),
                              device_ids=train_config.device_ids)
        # net = physformer.ViT_ST_ST_Compact3_TDC_gra_sharp()
        net = net.to(train_config.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        print("Training...")
        train(net, optimizer, scheduler, train_iter, train_config)
        torch.save(net.state_dict(), path + os.sep + f"physformer_T160_fold{i + 1}.pt")
        # test
        net = net.to(test_config.device)
        print(f"Evaluating...")
        # MAE, RMSE, MAPE, R
        temp = test(net, test_iter, test_config)
        print(f"MAE: {temp[0]: .3f}\n"
              f"RMSE: {temp[1]: .3f}\n"
              f"MAPE: {temp[2]: .3f}\n"
              f"R: {temp[3]: .3f}")
        result.update(val=temp, n=1)
    print(f"Cross Validation:\n"
          f"MAE: {result.acc[0] / result.cnt[0]: .3f}\n"
          f"RMSE: {result.acc[1] / result.cnt[1]: .3f}\n"
          f"MAPE: {result.acc[2] / result.cnt[2]: .3f}\n"
          f"R: {result.acc[3] / result.cnt[3]: .3f}")


def train(net: nn.Module, optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          train_iter: data.DataLoader, train_config: running.TrainConfig):
    net.train()
    loss_fun1 = loss_function.NegPearson()
    loss_fun2 = loss_function.FreqLoss(T=train_config.T)
    train_loss = metric.Accumulate(4)  # for print
    progress_bar = tqdm(range(len(train_iter) * train_config.num_epochs))
    # a: temporal loss; b: frequency loss
    a_start = 0.1
    b_start = 1.0
    exp_a = 0.5
    exp_b = 5.0
    for epoch in range(train_config.num_epochs):
        print(f"Epoch {epoch + 1}...")
        for x, y, average_hr, Fs in train_iter:
            # to cuda
            x = x.to(train_config.device)
            y = y.to(train_config.device)
            average_hr = average_hr.to(train_config.device)
            Fs = Fs.to(train_config.device)
            # forward
            preds, Score1, Score2, Score3 = net(x, train_config.tau)
            rppg_loss = loss_fun1(preds, y)  # temporal loss
            dist_loss, freq_loss, mae_loss = loss_fun2(preds, average_hr, Fs)
            if epoch > 25:
                a = 0.05
                b = 5.0
            else:
                # exp descend
                a = a_start * math.pow(exp_a, epoch / 25.0)
                # exp ascend
                b = b_start * math.pow(exp_b, epoch / 25.0)
            # backward
            optimizer.zero_grad()
            loss = a * rppg_loss + b * (freq_loss + dist_loss)
            loss.backward()
            optimizer.step()
            # for print
            val = [rppg_loss.data, freq_loss.data, dist_loss.data, mae_loss.data]
            train_loss.update(val=val, n=1)
            progress_bar.update(1)
        # scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"****************************************************"
                  f"Epoch{epoch + 1}:\n"
                  f"temporal loss: {train_loss.acc[0] / train_loss.cnt[0]: .3f}\n"
                  f"CELoss: {train_loss.acc[1] / train_loss.cnt[1]: .3f}\n"
                  f"distribution loss: {train_loss.acc[2] / train_loss.cnt[2]: .3f}\n"
                  f"MAE: {train_loss.acc[3] / train_loss.cnt[3]: .3f}\n"
                  f"****************************************************")


def test(net: nn.Module, test_iter: data.DataLoader,
         test_config: running.TestConfig) -> list:
    net.eval()
    pred_phys = []
    label_phys = []
    progress_bar = tqdm(range(len(test_iter)))
    for x, y, average_hr, Fs in test_iter:
        x = x.to(test_config.device)
        # y = y.to(test_config.device)
        average_hr = average_hr.to(test_config.device)
        Fs = Fs.to(test_config.device)

        preds, Score1, Score2, Score3 = net(x, test_config.tau)

        for i in range(len(x)):
            # 需要转换为 numpy 逐样本计算,
            if test_config.post == "fft":
                pred_temp = postprocess.fft_physiology(preds[i].detach().cpu().numpy(),
                                                       Fs=float(Fs[i]), diff=False,
                                                       detrend_flag=False).reshape(-1)
            else:
                pred_temp = postprocess.peak_physiology(preds[i].detach().cpu().numpy(),
                                                        Fs=float(Fs[i]), diff=False,
                                                        detrend_flag=False).reshape(-1)
            label_temp = average_hr[i].cpu().numpy().reshape(-1)
            pred_phys.append(pred_temp)
            label_phys.append(label_temp)
        progress_bar.update(1)
    pred_phys = np.asarray(pred_phys)
    label_phys = np.asarray(label_phys)

    return metric.cal_metric(pred_phys, label_phys)


cross_validation(5, "./saved", running.TrainConfig, running.TestConfig)
"""net = physformer.ViT_ST_ST_Compact3_TDC_gra_sharp()
net.load_state_dict(torch.load("./saved/physformer_T160_fold1.pt"))
test_config = running.TestFold1
net.to(test_config.device)
test_set = vipl_hr.VIPL_HR(test_config)
test_iter = data.DataLoader(test_set, batch_size=test_config.batch_size,
                            shuffle=False)
temp = test(net, test_iter, test_config)
print(f"MAE: {temp[0]: .3f}\n"
      f"RMSE: {temp[1]: .3f}\n"
      f"MAPE: {temp[2]: .3f}\n"
      f"R: {temp[3]: .3f}")"""
