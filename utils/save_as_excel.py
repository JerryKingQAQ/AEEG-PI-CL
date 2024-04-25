# -*- coding = utf-8 -*-
# @File : save_as_excel.py
# @Software : PyCharm
import os

import pandas as pd


def save_as_excel(data, args):
    # 创建一个DataFrame对象
    df = pd.DataFrame({"top1": data['top1'], "top5": data['top5']})
    save_path = "excel/{}/{}_{}_{}_{}.xlsx".format(args["cls_model"], args["cls_model"], args["backbone_type"],
                                                   args['init_cls'],
                                                   args['increment'])

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # 将DataFrame保存为Excel文件
    df.to_excel(save_path, index=False)
