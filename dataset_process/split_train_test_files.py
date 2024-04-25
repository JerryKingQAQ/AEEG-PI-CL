# -*- coding = utf-8 -*-
# @File : split_train_test_files.py
# @Software : PyCharm
import os

import numpy as np
import torch

from utils import custom_sort

emotions = ['anger', 'disgust', 'fear', 'sadness',
            'neutral', 'amusement', 'inspiration',
            'joy', 'tenderness']

emo_data_all = {'anger': [], 'disgust': [], 'fear': [], 'sadness': [],
                'neutral': [], 'amusement': [], 'inspiration': [],
                'joy': [], 'tenderness': []}

id_labels_all = {'anger': [], 'disgust': [], 'fear': [], 'sadness': [],
                 'neutral': [], 'amusement': [], 'inspiration': [],
                 'joy': [], 'tenderness': []}


def split_train_test_files(cross_trial=True):
    files_path = '../data/subject_preprocess'
    dataset_save_path = '../data/TsinghuaPIDatasets/'
    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)

    if cross_trial:
        base_path = os.path.join(dataset_save_path, 'cross_trial')
    else:
        base_path = os.path.join(dataset_save_path, 'not_cross_trial')
    train_data_path = os.path.join(base_path, 'train')
    test_data_path = os.path.join(base_path, 'test')
    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(test_data_path, exist_ok=True)

    files = os.listdir(files_path)
    files = sorted(files, key=custom_sort)

    for file in files:
        data = np.load(os.path.join(files_path, file), allow_pickle=True).item()
        for emotion in emotions:
            emo_data_all[emotion].append(data[emotion])
            id_labels_all[emotion].append(data['id'])

    # 将所有数据转化为numpy 划分训练集测试集
    for emotion in emotions:
        train_data = np.empty((0, 32, 375, 6))
        test_data = np.empty((0, 32, 375, 6))

        emo_data_all[emotion] = np.array(emo_data_all[emotion])
        id_labels_all[emotion] = np.array(id_labels_all[emotion])

        for i in range(emo_data_all[emotion].shape[0]):
            for j in range(emo_data_all[emotion].shape[1]):
                temp_data = np.expand_dims(emo_data_all[emotion][i, j], axis=0)
                if cross_trial:
                    if j < 20:  # 跨trial 前两个trial做训练集 最后一个trial做测试集
                        train_data = np.concatenate((train_data, temp_data))
                    else:
                        test_data = np.concatenate((test_data, temp_data))
                else:
                    if j % 3 == 0:  # 非跨trial train:test = 2:1
                        test_data = np.concatenate((test_data, temp_data))

                    else:
                        train_data = np.concatenate((train_data, temp_data))

        train_labels = id_labels_all[emotion].repeat(20)
        test_labels = id_labels_all[emotion].repeat(10)

        print(f'Emotion Data is {emotion}, Emotion Data Shape is {emo_data_all[emotion].shape}\n'
              f'ID Data Shape is {id_labels_all[emotion].shape}')

        print(f'Train Data Shape is {train_data.shape}, Test Data Shape is {test_data.shape}\n'
              f'Train Labels Shape is {train_labels.shape}, Test Labels Shape is {test_labels.shape}')

        train_data_file = os.path.join(train_data_path, f'{emotion}_data.pth')
        train_labels_file = os.path.join(train_data_path, f'{emotion}_labels.pth')
        test_data_file = os.path.join(test_data_path, f'{emotion}_data.pth')
        test_labels_file = os.path.join(test_data_path, f'{emotion}_labels.pth')

        torch.save(train_data, train_data_file)
        torch.save(train_labels, train_labels_file)
        torch.save(test_data, test_data_file)
        torch.save(test_labels, test_labels_file)


if __name__ == '__main__':
    split_train_test_files(cross_trial=True)
