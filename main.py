import argparse
import copy
import json
import logging
import os.path
import random

import pandas as pd

from trainer import train


def main(train_args):
    result = train(train_args)
    return result


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default=f'exps/models/exper_1_1.json',
                        help='Json file of settings.')

    # exps/jointed/exper_7_1.json
    # exps/exper_1_1.json
    # exps/exper_2_6.json
    # exps/exper_visualization.json
    # exps/exper_time_cpu.json
    # exps/cross_dataset/exper_7_1.json

    return parser


def make_excel_map(args):
    my_dict = {}
    if args["cls_model"] == ["joint"]:
        start = 1
        end = 1
        step = 1
    else:
        start = args['init_cls']
        end = args['num_classes']
        step = args['increment']
    for i in range(start, end + 1, step):
        my_dict[str(i)] = []
    return my_dict


if __name__ == '__main__':
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    model_list = copy.deepcopy(args['cls_model'])
    backbone_list = copy.deepcopy(args['backbone_type'])
    emotion_list = copy.deepcopy(args['emotion'])
    repeat = args['repeat']
    excel_map = make_excel_map(args)

    for model in model_list:
        for backbone in backbone_list:
            for emotion in emotion_list:

                top1_df = pd.DataFrame(excel_map)
                top5_df = pd.DataFrame(excel_map)

                for i in range(repeat):
                    train_args = args
                    train_args['round'] = i
                    train_args['cls_model'] = model
                    train_args['backbone_type'] = backbone
                    train_args['emotion'] = emotion
                    if train_args['random_seed']:  # if seed equal -1, get into a random state
                        train_args['seed'] = random.randint(0, 2 ** 32 - 1)
                    result = main(train_args)
                    result[f'experiment {i}'] = i

                    # add top1 result
                    top1_df.loc[len(top1_df)] = result["cnn_curve"]['top1']

                    # add top5 result
                    top5_df.loc[len(top5_df)] = result["cnn_curve"]['top5']

                top1_save_path = f"excel/{train_args['dataset']}/{train_args['frequency']}_{emotion}_{model}_{backbone}" \
                                 f"_{train_args['init_cls']}_{train_args['increment']}_top1.xlsx"

                top5_save_path = f"excel/{train_args['dataset']}/{train_args['frequency']}_{emotion}_{model}_{backbone}" \
                                 f"_{train_args['init_cls']}_{train_args['increment']}_top5.xlsx"

                if not os.path.exists(os.path.dirname(top1_save_path)):
                    os.makedirs(os.path.dirname(top1_save_path))

                if not os.path.exists(os.path.dirname(top5_save_path)):
                    os.makedirs(os.path.dirname(top5_save_path))

                # save result
                top1_df.to_excel(top1_save_path, index=True)
                top5_df.to_excel(top5_save_path, index=True)
