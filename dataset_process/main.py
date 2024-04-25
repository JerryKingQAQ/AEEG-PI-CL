# -*- coding = utf-8 -*-
# @File : main.py
# @Software : PyCharm
# @Author : Jiarui Jin
from dataset_process.preprocess import preprocess_main
from dataset_process.split_train_test_files import split_train_test_files

if __name__ == '__main__':
    files_path = 'path/to/THU-EP/'
    preprocess_main(files_path)
    split_train_test_files(cross_trial=True)