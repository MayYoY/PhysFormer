import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from fastai.losses import FocalLoss


class NegPearson(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(NegPearson, self).__init__()
        assert reduction in ["mean", "sum", "none"], "Unsupported reduction type!"
        self.reduction = reduction

    def forward(self, preds, labels):
        sum_x = torch.sum(preds, dim=1)
        sum_y = torch.sum(labels, dim=1)
        sum_xy = torch.sum(labels * preds, dim=1)
        sum_x2 = torch.sum(preds ** 2, dim=1)
        sum_y2 = torch.sum(labels ** 2, dim=1)
        T = preds.shape[1]
        # 防止对负数开根号
        denominator = (T * sum_x2 - sum_x ** 2) * (T * sum_y2 - sum_y ** 2)
        for i in range(len(denominator)):
            denominator[i] = max(denominator[i], 1e-8)
        loss = 1 - ((T * sum_xy - sum_x * sum_y) / (torch.sqrt(denominator)) + 1e-8)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def normal_sampling(mean, label_k, std=1.0):
    """
    \frac{1}{\sqrt{2\pi} * \sigma} * \exp(-\frac{(x - \mu)^2}{2 * \sigma^2})
    :param mean:
    :param label_k:
    :param std: = 2
    :return:
    """
    return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)


class FreqLoss(nn.Module):
    def __init__(self, T=300, delta=3, reduction="mean"):
        """
        :param T: 序列长度
        :param delta: 信号带宽, 带宽外的认为是噪声, 验证阶段的 delta 为 60 * 0.1
        :param reduction:
        """
        super(FreqLoss, self).__init__()
        self.T = T
        self.delta = delta
        self.low_bound = 40
        self.high_bound = 180
        # for DFT
        self.bpm_range = torch.arange(self.low_bound, self.high_bound,
                                      dtype=torch.float) / 60.
        self.two_pi_n = Variable(2 * math.pi * torch.arange(0, self.T, dtype=torch.float))
        self.hanning = Variable(torch.from_numpy(np.hanning(self.T)).type(torch.FloatTensor),
                                requires_grad=True).view(1, -1)  # 1 x N
        # criterion
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_diverse = nn.KLDivLoss(reduction="sum")
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, wave, labels, fps):
        """
        DFT: F(**k**) = \sum_{n = 0}^{N - 1} f(n) * \exp{-j2 \pi n **k** / N}
        :param wave: predict ecg  B x N
        :param labels: heart rate B,
        :param fps: B,
        :return:
        """
        # 多 GPU 训练下, 确保同一 device
        self.bpm_range = self.bpm_range.to(wave.device)
        self.two_pi_n = self.two_pi_n.to(wave.device)
        self.hanning = self.hanning.to(wave.device)

        # DFT
        B = wave.shape[0]
        # DFT 中的 k = N x fk / fs, x N 与 (-j2 \pi n **k** / N) 抵消
        k = self.bpm_range[None, :] / fps[:, None]
        k = k.view(B, -1, 1)  # B x range x 1
        # 汉宁窗
        preds = wave * self.hanning  # B x N
        preds = preds.view(B, 1, -1)  # B x 1 x N
        # 2 \pi n
        temp = self.two_pi_n.repeat(B, 1)
        temp = temp.view(B, 1, -1)  # B x 1 x N
        # B x range
        complex_absolute = torch.sum(preds * torch.sin(k * temp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(k * temp), dim=-1) ** 2
        # 归一化
        norm_t = torch.ones(B, device=wave.device) / (torch.sum(complex_absolute, dim=1) + 1e-8)
        norm_t = norm_t.view(-1, 1)  # B x 1
        complex_absolute = complex_absolute * norm_t  # B x range
        # 平移区间 [40, 180] -> [0, 140]
        gts = labels.clone()
        gts -= self.low_bound
        gts[gts.le(0)] = 0
        gts[gts.ge(139)] = 139
        gts = gts.type(torch.long).view(B)

        # 预测心率
        _, whole_max_idx = complex_absolute.max(1)
        print(whole_max_idx + 40)
        freq_loss = self.cross_entropy(complex_absolute, gts)

        # KL loss
        gts_distribution = []
        for gt in gts:
            temp = [normal_sampling(int(gt), i, std=1.) for i in range(140)]
            temp = [i if i > 1e-15 else 1e-15 for i in temp]  # 替 0
            gts_distribution.append(temp)
        gts_distribution = torch.tensor(gts_distribution, device=wave.device)
        freq_distribution = F.log_softmax(complex_absolute, dim=-1)
        dist_loss = self.kl_diverse(freq_distribution, gts_distribution) / B

        # MAE loss
        mae_loss = self.l1_loss(whole_max_idx.type(torch.float), gts.type(torch.float))

        return dist_loss, freq_loss, mae_loss

