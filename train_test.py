import torch
import torch.nn as nn
from torch.utils import data
import os
import math
import random
import numpy as np
from tqdm.auto import tqdm

from dataset import vipl_hr
from evaluate import metric, postprocess
from configs import running
from model import loss_function, physformer


def merge_clips(x):
    sort_x = sorted(x.items(), key=lambda x: x[0])
    sort_x = [i[1] for i in sort_x]
    # sort_x = torch.cat(sort_x, dim=0)
    sort_x = np.concatenate(sort_x, axis=0)
    return sort_x.reshape(-1)


def cross_validation(folds, path, train_config, test_config, methods=None,
                     mode="Train", model_path=""):
    """
    受试者独立交叉验证
    :param folds:
    :param path: for saving models
    :param train_config:
    :param test_config:
    :param methods:
    :param mode: Train or Test
    :param model_path: if only test, pretrained model should be provide
    :return:
    """
    if methods is None:
        methods = ["Mean", "Std", "MAE", "RMSE", "MAPE", "R"]
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
        net = physformer.ViT_ST_ST_Compact3_TDC_gra_sharp()
        if mode == "Train":
            net = net.to(train_config.device)
            # Adam optimizer and the initial learning rate and weight
            # decay are 1e-4 and 5e-5, respectively
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            print("Training...")
            train(net, optimizer, scheduler, train_iter, train_config, test_iter, test_config, path)
            torch.save(net.state_dict(), path + os.sep + f"physformer.pt")
        else:
            assert model_path, "Pretrained model is required!"
            net.load_state_dict(torch.load(model_path))
        # test
        net = net.to(test_config.device)
        print(f"Evaluating...")
        # MAE, RMSE, MAPE, R
        temp = test(net, test_iter, test_config)
        print(f"Mean: {temp[0]: .3f}\n"
              f"Std: {temp[1]: .3f}\n"
              f"MAE: {temp[2]: .3f}\n"
              f"RMSE: {temp[3]: .3f}\n"
              f"MAPE: {temp[4]: .3f}\n"
              f"R: {temp[5]: .3f}")
        result.update(val=temp, n=1)
    print(f"Cross Validation:\n"
          f"Mean: {result.acc[0] / result.cnt[0]: .3f}\n"
          f"Std: {result.acc[1] / result.cnt[1]: .3f}\n"
          f"MAE: {result.acc[2] / result.cnt[2]: .3f}\n"
          f"RMSE: {result.acc[3] / result.cnt[3]: .3f}\n"
          f"MAPE: {result.acc[4] / result.cnt[4]: .3f}\n"
          f"R: {result.acc[5] / result.cnt[5]: .3f}")


def train(net: nn.Module, optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          train_iter: data.DataLoader, train_config: running.TrainConfig,
          test_iter: data.DataLoader, test_config: running.TestConfig, path):
    net.train()
    net.to(train_config.device)
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
        train_loss.reset()
        net.train()
        net.to(train_config.device)
        print(f"Epoch {epoch + 1}...")
        for x, y, average_hr, Fs, _, _ in train_iter:
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

            nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=2)

            optimizer.step()
            # for print
            val = [rppg_loss.data, freq_loss.data, dist_loss.data, mae_loss.data]
            train_loss.update(val=val, n=1)
            progress_bar.update(1)
        scheduler.step()
        print(f"****************************************************\n"
              f"Epoch{epoch + 1}:\n"
              f"temporal loss: {train_loss.acc[0] / train_loss.cnt[0]: .3f}\n"
              f"CELoss: {train_loss.acc[1] / train_loss.cnt[1]: .3f}\n"
              f"distribution loss: {train_loss.acc[2] / train_loss.cnt[2]: .3f}\n"
              f"MAE: {train_loss.acc[3] / train_loss.cnt[3]: .3f}\n"
              f"****************************************************")
        torch.save(net.state_dict(), path + os.sep + f"physformer_epoch{epoch + 1}.pt")


def test(net: nn.Module, test_iter: data.DataLoader,
         test_config: running.TestConfig) -> list:
    net.eval()
    net.to(test_config.device)
    predictions = dict()
    labels = dict()
    frame_rates = dict()
    progress_bar = tqdm(range(len(test_iter)))
    for x, y, _, Fs, subjects, chunks in test_iter:
        x = x.to(test_config.device)
        y = y.to(test_config.device)
        Fs = Fs.to(test_config.device)

        preds, Score1, Score2, Score3 = net(x, test_config.tau)

        for i in range(len(x)):
            file_name = subjects[i]
            chunk_idx = chunks[i]
            if file_name not in predictions.keys():
                predictions[file_name] = dict()
                labels[file_name] = dict()
                frame_rates[file_name] = float(Fs[i])
            predictions[file_name][chunk_idx] = preds[i].detach().cpu().numpy()
            labels[file_name][chunk_idx] = y[i].detach().cpu().numpy()
        progress_bar.update(1)
    pred_phys = []
    label_phys = []
    bar = tqdm(range(len(predictions.keys())))
    # 合并同一视频的预测 clip
    for file_name in predictions.keys():
        pred_temp = merge_clips(predictions[file_name])
        label_temp = merge_clips(labels[file_name])
        if test_config.post == "fft":
            pred_temp = postprocess.fft_physiology(pred_temp, Fs=frame_rates[file_name],
                                                   diff=test_config.diff,
                                                   detrend_flag=test_config.detrend).reshape(-1)
            label_temp = postprocess.fft_physiology(label_temp, Fs=frame_rates[file_name],
                                                    diff=test_config.diff,
                                                    detrend_flag=test_config.detrend).reshape(-1)
        else:
            pred_temp = postprocess.peak_physiology(pred_temp, Fs=frame_rates[file_name],
                                                    diff=test_config.diff,
                                                    detrend_flag=test_config.detrend).reshape(-1)
            label_temp = postprocess.peak_physiology(label_temp, Fs=frame_rates[file_name],
                                                     diff=test_config.diff,
                                                     detrend_flag=test_config.detrend).reshape(-1)
        pred_phys.append(pred_temp)
        label_phys.append(label_temp)
        bar.update(1)
    pred_phys = np.asarray(pred_phys)
    label_phys = np.asarray(label_phys)

    return metric.cal_metric(pred_phys, label_phys)


def fixSeed(seed: int):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu
    # torch.backends.cudnn.deterministic = True  # 会大大降低速度
    torch.backends.cudnn.benchmark = True  # False会确定性地选择算法，会降低性能
    torch.backends.cudnn.enabled = True  # 增加运行效率，默认就是True
    torch.manual_seed(seed)


fixSeed(42)
cross_validation(1, "./saved", running.TrainConfig, running.TestConfig)