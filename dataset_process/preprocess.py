# -*- coding = utf-8 -*-
# @File : preprocess.py
# @Software : PyCharm
import fnmatch
import os

import h5py
import numpy as np

from utils import custom_sort, downsample_process, make_labels, split_data


def preprocess_main(files_path):
    files_save_path = '../data/subject_preprocess/'

    if not os.path.exists(files_save_path):
        os.makedirs(files_save_path)

    files = os.listdir(files_path)
    files = [f for f in files if fnmatch.fnmatch(f, '*.mat')]
    files = sorted(files, key=custom_sort)

    for file in files:
        data = h5py.File(os.path.join(files_path, file), 'r')
        # shape:(time points,channels,stimuli,bands)
        np_data = np.array(data['data'])

        # downsample
        sub_data = downsample_process(np_data)

        # split
        sub_data = split_data(sub_data)

        # add labels
        sub_data = make_labels(file, sub_data)

        print(sub_data['id'])
        file_name = file.split('.')[0]
        np.save(files_save_path + f'{file_name}.npy', sub_data)
        print(f'File {file_name} preprocessed.')
        data.close()


if __name__ == '__main__':
    preprocess_main()
