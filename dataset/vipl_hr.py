import cv2 as cv
import numpy as np
import pandas as pd
import torch
import os
import glob
import re

from tqdm.auto import tqdm
from scipy import interpolate, io
from torch.utils import data

from . import utils


# TODO: 本次修改了 HR 序列的读入, 会 truncate hrs or frames; 同时选择不放大脸部区域
class FramePreprocess:
    def __init__(self, config):
        self.config = config
        # [p1, p2, ..., pn]
        self.dirs = glob.glob(self.config.input_path + os.sep + "data" + os.sep + "*")
        self.folds = self.get_fold()

    def get_fold(self):
        ret = {}
        files = glob.glob(self.config.input_path + os.sep + "fold" + os.sep + "*.mat")
        for f in files:
            i = int(f[-5])
            temp = io.loadmat(f)[f"fold{i}"][0]  # subject_idx of fold(i + 1)
            for idx in temp:
                ret[idx] = i
        return ret

    def read_process(self):
        file_num = len(self.dirs)
        progress_bar = tqdm(list(range(file_num)))
        csv_info = {"input_files": [], "wave_files": [], "start": [], "end": [],
                    "fold": [], "average_HR": [], "Fs": [], "task": [], "source": []}
        for pi in self.dirs:  # i_th subject
            p_idx = re.findall("p(\d\d?\d?)", pi)[0]  # p1, p2, ..., p10, p100, p101
            tasks = glob.glob(pi + os.sep + "*")  # [v1(, v1-2), v2, ...]
            for ti in tasks:
                if not re.findall("v(\d-\d)", ti):
                    t_idx = re.findall("v(\d)", ti)[0]
                else:
                    t_idx = re.findall("v(\d-\d)", ti)[0]  # v1-2
                sources = glob.glob(ti + os.sep + "*")  # [source1, source2, ...]
                for si in sources:
                    s_idx = re.findall("source(\d)", si)[0]  # source_i
                    filename = f"p{p_idx}_v{t_idx}_source{s_idx}"
                    clip_range, Fs, end_time = self.read_video(si, filename)  # T,
                    # HR 序列, 需要根据视频长度截取 (过长截取 HR, 过短截取视频)
                    hrs = self.read_hrs(si, end_time)
                    if round(len(hrs) * Fs) < len(clip_range):
                        clip_range = clip_range[: round(len(hrs) * Fs)]
                    if len(clip_range) < self.config.CHUNK_LENGTH:  # 视频太短, 丢弃
                        continue
                    # 对 BVP 信号插值, 对齐
                    waves = self.read_wave(si)  # T_w,
                    fun = interpolate.CubicSpline(range(len(waves)), waves)
                    x_new = np.linspace(0, len(waves) - 1, num=len(clip_range))
                    gts = fun(x_new)  # T
                    # chunk, n x len, n x len
                    frames_clips, gts_clips = self.preprocess(clip_range, gts)
                    # 命名信息
                    single_info = {"filename": filename,
                                   "fold": self.folds[int(p_idx)], "Fs": Fs}
                    # input_list, wave_list, start_list, end_list, fold_list, HR_list, Fs_list
                    temp = self.save(frames_clips, gts_clips, hrs, single_info)
                    csv_info["wave_files"] += temp[0]
                    csv_info["start"] += temp[1]
                    csv_info["end"] += temp[2]
                    csv_info["average_HR"] += temp[3]
                    # clips 间相同的信息
                    N = len(gts_clips)
                    csv_info["input_files"] += [self.config.img_cache + os.sep +
                                                f"{single_info['filename']}"] * N
                    csv_info["fold"] += [single_info["fold"]] * N
                    csv_info["Fs"] += [single_info["Fs"]] * N
                    csv_info["task"] += [int(t_idx[0])] * N
                    csv_info["source"] += [int(s_idx)] * N
            progress_bar.update(1)

        csv_info = pd.DataFrame(csv_info)
        csv_info.to_csv(self.config.record_path, index=False)

    def save(self, frames_clips: np.array, gts_clips: np.array,
             hrs: np.ndarray, single_info: dict):
        """Saves the preprocessing data."""
        # 生成对应的文件夹
        os.makedirs(self.config.gt_cache, exist_ok=True)
        wave_list = []  # 标签路径
        start_list = []  # clip 范围
        end_list = []
        HR_list = []  # 平均 HR
        step = len(hrs) // len(gts_clips)  # 根据 clip_num 分割 HR 序列
        # 保存文件
        for i in range(len(gts_clips)):
            HR_list.append(hrs[i * step: (i + 1) * step].mean())  # clip 的平均 HR
            # 保存处理好的 wave clip
            label_path = self.config.gt_cache + os.sep + f"{single_info['filename']}_label{i}.npy"
            np.save(label_path, gts_clips[i])
            wave_list.append(label_path)
            # 记录 clip 的范围
            start_list.append(frames_clips[i, 0])
            end_list.append(frames_clips[i, -1] + 1)
        return wave_list, start_list, end_list, HR_list

    def preprocess(self, clip_range, gts):
        """
        normalize / standardize
        :param clip_range: array, T,
        :param gts: array, T,
        """
        # 标签需要 standardize
        y = utils.standardize(gts[:])
        # 分块
        if self.config.DO_CHUNK:
            frames_clips, gts_clips = utils.chunk(clip_range, y, self.config.CHUNK_LENGTH)
        else:
            frames_clips = np.array([clip_range])  # n x len x H x W x C
            gts_clips = np.array([y])  # n x len

        return frames_clips, gts_clips

    def read_video(self, data_path, filename):
        """读取视频, 人脸检测, 保存帧; 返回帧下标, 帧率"""
        save_dir = self.config.img_cache + os.sep + filename
        if self.config.MODIFY:
            vid = cv.VideoCapture(data_path + os.sep + "video.avi")
            vid.set(cv.CAP_PROP_POS_MSEC, 0)  # 设置从 0 开始读取
            ret, frame = vid.read()
            frames = list()
            while ret:
                frame = cv.cvtColor(np.array(frame), cv.COLOR_BGR2RGB)
                frame = np.asarray(frame)
                frame[np.isnan(frame)] = 0
                frames.append(frame)
                ret, frame = vid.read()

            frames = np.asarray(frames)

            # 人脸检测并截取
            frames = utils.resize(frames, self.config.DYNAMIC_DETECTION,
                                  self.config.DYNAMIC_DETECTION_FREQUENCY,
                                  self.config.W, self.config.H,
                                  self.config.LARGE_FACE_BOX,
                                  self.config.CROP_FACE,
                                  self.config.LARGE_BOX_COEF).astype(np.uint8)
            # 保存处理好的帧
            for i, frame in enumerate(frames):
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                os.makedirs(save_dir, exist_ok=True)
                cv.imwrite(save_dir + os.sep + f"{i}.png", frame)
            T = len(frames)
        else:
            T = len(glob.glob(save_dir + os.sep + "*.png"))
        # for return
        clips_range = np.arange(T)
        if data_path[-1] == "2":
            Fs = 30
            # 计算视频结束时间
            end_time = round(T / Fs)
        else:
            time_record = np.loadtxt(data_path + os.sep + "time.txt")
            bound = min(len(time_record) - 1, T - 1)
            Fs = T * 1000 / time_record[bound]
            end_time = round(time_record[bound] / 1000)
        return clips_range, Fs, end_time

    @staticmethod
    def read_wave(data_path):
        """
        读取 bvp 信号
        :param data_path:
        :return np.array T, ; bvp
        """
        waves = pd.read_csv(data_path + os.sep + "wave.csv")["Wave"].values
        return waves

    @staticmethod
    def read_hrs(data_path, end_time):
        # The HR and SpO2 of the subject is recorded every second
        # 根据结束时间截取 HR 序列
        hrs = pd.read_csv(data_path + os.sep + "gt_HR.csv")["HR"].values[: end_time]
        return hrs


class VIPL_HR(data.Dataset):
    def __init__(self, config):
        super(VIPL_HR, self).__init__()
        record = pd.read_csv(config.record)
        self.config = config
        self.input_files = []
        self.wave_files = []
        self.starts = []
        self.ends = []
        self.average_hrs = []
        self.Fs = []
        for i in range(len(record)):
            if self.isValid(record, i):
                self.input_files.append(record.loc[i, "input_files"])
                self.wave_files.append(record.loc[i, "wave_files"])
                self.starts.append(record.loc[i, "start"])
                self.ends.append(record.loc[i, "end"])
                self.average_hrs.append(record.loc[i, "average_HR"])
                self.Fs.append(record.loc[i, "Fs"])

    def isValid(self, record, idx):
        flag = True
        if self.config.folds:
            flag &= record.loc[idx, "fold"] in self.config.folds
        if self.config.tasks:
            flag &= record.loc[idx, "task"] in self.config.tasks
        if self.config.sources:
            flag &= record.loc[idx, "source"] in self.config.sources
        return flag

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # TODO: 数据增强, 随机 mask
        x_path = self.input_files[idx]
        x = []
        for i in range(self.starts[idx], self.ends[idx]):
            temp = cv.imread(x_path + os.sep + f"{i}.png")
            # H x W x C
            x.append(cv.cvtColor(temp, cv.COLOR_BGR2RGB))
        x = utils.normalize_frame(np.asarray(x, dtype=np.double))  # 归一化
        x = torch.from_numpy(x).permute(3, 0, 1, 2)  # T x H x W x C -> C x T x H x W
        # 读取标签信息
        y_path = self.wave_files[idx]
        y = torch.from_numpy(np.load(y_path))  # T,
        average_hr = torch.tensor([self.average_hrs[idx]])
        # 帧率
        Fs = torch.tensor([self.Fs[idx]])
        # torchvision.transforms.RandomHorizontalFlip
        if self.config.trans is not None:
            x = self.config.trans(x)

        # 视频段信息, 用于 merge
        item_path = self.wave_files[idx]
        item_path_filename = item_path.split('/')[-1]
        split_idx = item_path_filename.rindex('_')  # rindex 检测最后一个 _
        file_name = item_path_filename[:split_idx]
        chunk_idx = item_path_filename[split_idx + 6:].split('.')[0]

        return x.float(), y.float(), average_hr.float(), Fs.float(), file_name, chunk_idx
