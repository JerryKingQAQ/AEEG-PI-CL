# -*- coding = utf-8 -*-
# @File : utils.py
# @Software : PyCharm

import numpy as np
from scipy.signal import resample

target_emotion = {'anger': [0, 1, 2], 'disgust': [3, 4, 5], 'fear': [6, 7, 8], 'sadness': [9, 10, 11],
                  'neutral': [12, 13, 14, 15], 'amusement': [16, 17, 18], 'inspiration': [19, 20, 21],
                  'joy': [22, 23, 24], 'tenderness': [25, 26, 27]}


def split_data(sub_data):
    tw = 375
    new_sub_data = {}
    for emotion in sub_data.keys():
        split_emo_data = []
        for i in range(sub_data[emotion].shape[0]):
            if i == 3:  # neutral取3个trial
                continue
            start = 0
            for step in range(sub_data[emotion].shape[2] // tw):
                temp_data = sub_data[emotion][i, :, start:start + tw, :]
                split_emo_data.append(temp_data)
                start += tw

        split_emo_data = np.array(split_emo_data)

        new_sub_data[emotion] = split_emo_data
    return new_sub_data


def make_labels(file, sub_data):
    id = file.split('_')[1]
    id = id.split('.')[0]
    id = int(id) - 1
    sub_data['id'] = id
    return sub_data


def custom_sort(file_name):
    # 提取文件名中下划线后的数字部分，并将其转换为整数
    parts = file_name.split('_')[1]
    num = parts.split('.')[0]
    return int(num)


def downsample(data):
    original_sampling_rate = 250
    target_sampling_rate = 125

    original_length = data.shape[2]
    target_length = int(original_length * target_sampling_rate / original_sampling_rate)
    resampled_data = np.empty((data.shape[0], data.shape[1], target_length, data.shape[3]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[3]):
                resampled_data[i, j, :, k] = resample(data[i, j, :, k], target_length)

    return resampled_data


def downsample_process(data):
    sub_data = {}
    # shape:(stimuli,channels,time points,bands)
    data = np.transpose(data, axes=(2, 1, 0, 3))

    for emotion in target_emotion.keys():
        emotion_data = data[target_emotion[emotion], :, :, :]
        downsample_emotion_data = downsample(emotion_data)
        sub_data[emotion] = downsample_emotion_data
    return sub_data
