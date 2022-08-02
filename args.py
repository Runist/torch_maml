# -*- coding: utf-8 -*-
# @File : args.py
# @Author: Runist
# @Time : 2022/3/29 9:46
# @Software: PyCharm
# @Brief: Code argument parser

import argparse
import warnings
import os
import torch
import sys
sys.path.append(os.getcwd())

from core.helper import seed_torch

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
parser.add_argument('--train_data_dir', type=str,
                    default="./data/Omniglot/images_background/",
                    help='The directory containing the train image data.')
parser.add_argument('--val_data_dir', type=str,
                    default="./data/Omniglot/images_evaluation/",
                    help='The directory containing the validation image data.')
parser.add_argument('--summary_path', type=str,
                    default="./summary",
                    help='The directory of the summary writer.')

parser.add_argument('--task_num', type=int, default=32,
                    help='Number of task per train batch.')
parser.add_argument('--val_task_num', type=int, default=16,
                    help='Number of task per test batch.')
parser.add_argument('--num_workers', type=int, default=12, help='The number of torch dataloader thread.')

parser.add_argument('--epochs', type=int, default=150,
                    help='The training epochs.')
parser.add_argument('--inner_lr', type=float, default=0.04,
                    help='The learning rate of of the support set.')
parser.add_argument('--outer_lr', type=float, default=0.001,
                    help='The learning rate of of the query set.')

parser.add_argument('--n_way', type=int, default=5,
                    help='The number of class of every task.')
parser.add_argument('--k_shot', type=int, default=1,
                    help='The number of support set image for every task.')
parser.add_argument('--q_query', type=int, default=1,
                    help='The number of query set image for every task.')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
seed_torch(1206)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
