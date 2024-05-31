# -*- coding: utf-8 -*-
# @File : main.py
# @Software : PyCharm
import argparse
from dataset_process.preprocess import preprocess_main
from dataset_process.split_train_test_files import split_train_test_files


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess and split dataset')
    parser.add_argument('--files_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--cross_trial', default='True', help='Flag for cross_trial in split_train_test_files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess_main(args.files_path)
    split_train_test_files(cross_trial=args.cross_trial)
