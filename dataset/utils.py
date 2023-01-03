import numpy as np
import pandas as pd
import cv2 as cv
from mtcnn import MTCNN
from math import ceil


def resize(frames, dynamic_det, det_length,
           w, h, larger_box, crop_face, larger_box_size):
    """
    :param frames:
    :param dynamic_det: 是否动态检测
    :param det_length: the interval of dynamic detection
    :param w:
    :param h:
    :param larger_box: whether to enlarge the detected region.
    :param crop_face:  whether to crop the frames.
    :param larger_box_size:
    """
    if dynamic_det:
        det_num = ceil(len(frames) / det_length)  # 检测次数
    else:
        det_num = 1
    face_region = []
    # 获取人脸区域
    detector = MTCNN()
    for idx in range(det_num):
        if crop_face:
            face_region.append(facial_detection(detector, frames[det_length * idx],
                                                larger_box, larger_box_size))
        else:  # 不截取
            face_region.append([0, 0, frames.shape[1], frames.shape[2]])
    face_region_all = np.asarray(face_region, dtype='int')
    # resize_frames = np.zeros((frames.shape[0], h, w, 3))  # T x H x W x 3
    resize_frames = []

    # 截取人脸并 resize
    for i in range(len(frames)):
        frame = frames[i]
        # 选定人脸区域
        if dynamic_det:
            reference_index = i // det_length
        else:
            reference_index = 0
        if crop_face:
            face_region = face_region_all[reference_index]
            frame = frame[max(face_region[1], 0):min(face_region[3], frame.shape[0]),
                          max(face_region[0], 0):min(face_region[2], frame.shape[1])]
        if w > 0 and h > 0:
            resize_frames.append(cv.resize(frame, (w + 4, h + 4),
                                           interpolation=cv.INTER_CUBIC)[2: w + 2, 2: h + 2, :])
        else:
            resize_frames.append(frame)
    if w > 0 and h > 0:
        return np.asarray(resize_frames)
    else:  # list
        return resize_frames


def facial_detection(detector, frame, larger_box=False, larger_box_size=1.0):
    """
    利用 MTCNN 检测人脸区域
    :param detector:
    :param frame:
    :param larger_box: 是否放大 bbox, 处理运动情况
    :param larger_box_size:
    """
    face_zone = detector.detect_faces(frame)
    if len(face_zone) < 1:
        print("Warning: No Face Detected!")
        return [0, 0, frame.shape[0], frame.shape[1]]
    if len(face_zone) >= 2:
        print("Warning: More than one faces are detected(Only cropping the biggest one.)")
    result = face_zone[0]['box']
    h = result[3]
    w = result[2]
    result[2] += result[0]
    result[3] += result[1]
    if larger_box:
        print("Larger Bounding Box")
        result[0] = round(max(0, result[0] + (1. - larger_box_size) / 2 * w))
        result[1] = round(max(0, result[1] + (1. - larger_box_size) / 2 * h))
        result[2] = round(max(0, result[0] + (1. + larger_box_size) / 2 * w))
        result[3] = round(max(0, result[1] + (1. + larger_box_size) / 2 * h))
    return result


def chunk(frames, gts, chunk_length, chunk_stride=-1):
    """Chunks the data into clips."""
    if chunk_stride < 0:
        chunk_stride = chunk_length
    # clip_num = (frames.shape[0] - chunk_length + chunk_stride) // chunk_stride
    frames_clips = [frames[i: i + chunk_length]
                    for i in range(0, frames.shape[0] - chunk_length + 1, chunk_stride)]
    bvps_clips = [gts[i: i + chunk_length]
                  for i in range(0, gts.shape[0] - chunk_length + 1, chunk_stride)]
    return np.array(frames_clips), np.array(bvps_clips)


def normalize_frame(frame):
    """[0, 255] -> [-1, 1]"""
    return (frame - 127.5) / 128


def standardize(data):
    """
    :param data:
    :return: (x - \mu) / \sigma
    """
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data


# Warning: 不现实, 上采样需要两倍帧率, 视频后半段无法增广
def temporal_sample(x, y, average_hr):
    """
    根据 HR 选择时序上采样 or 下采样
    temporally upsample and downsample the videos to generate extra training samples
    with extreme small or large HR values.
    the videos with HR values larger than 88 bpm would be temporally
    interpolated twice while those with HR smaller than 88 bpm
    are downsampled with sampling rate 2, to simulate half and
    doubled heart rate, respectively.
    :param x:
    :param y:
    :param average_hr:
    :return:
    """
    x_new = np.zeros_like(x)
    y_new = np.zeros_like(y)
    if average_hr > 88:  # 下采样
        average_hr /= 2
        for i in range(len(x)):
            if i % 2 == 0:
                x_new[:, i, :, :, :] = x[:, i // 2, :, :, :]  # C x T x H x W
                y_new[i] = y[i // 2]
            else:
                y_new[i] = y[i // 2] / 2 + y[i // 2 + 1] / 2
                x_new[:, i, :, :, :] = (x[:, i // 2, :, :, :] + x[:, i // 2 + 1, :, :, :]) / 2
    else:  # 上采样
        average_hr *= 2
        for i in range(len(x)):
            pass
    return x_new, y_new, average_hr
