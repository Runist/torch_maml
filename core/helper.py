# -*- coding: utf-8 -*-
# @File : helper.py
# @Author: Runist
# @Time : 2022/7/6 11:16
# @Software: PyCharm
# @Brief:
from net.maml import Classifier
from core.dataset import OmniglotDataset

import os
import torch
from torch import nn
import numpy as np
import random


def get_model(args, dev):
    """
    Get model.
    Args:
        args: ArgumentParser
        dev: torch dev

    Returns: model

    """
    model = Classifier(1, args.n_way).cuda()
    model.to(dev)

    return model


def get_dataset(args):
    """
    Get maml dataset.
    Args:
        args: ArgumentParser

    Returns: dataset

    """
    train_dataset = OmniglotDataset(args.train_data_dir, args.task_num,
                                    n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
    val_dataset = OmniglotDataset(args.val_data_dir, args.val_task_num,
                                  n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)

    return train_dataset, val_dataset


def seed_torch(seed):
    """
    Set all random seed
    Args:
        seed: random seed

    Returns: None

    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def remove_dir_and_create_dir(dir_name, is_remove=True):
    """
    Make new folder, if this folder exist, we will remove it and create a new folder.
    Args:
        dir_name: path of folder
        is_remove: if true, it will remove old folder and create new folder

    Returns: None

    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "create.")
    else:
        if is_remove:
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)
            print(dir_name, "create.")
        else:
            print(dir_name, "is exist.")