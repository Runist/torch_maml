# -*- coding: utf-8 -*-
# @File : dataset.py
# @Author: Runist
# @Time : 2022/7/6 10:38
# @Software: PyCharm
# @Brief:

from torch.utils.data.dataset import Dataset


class MAMLDataset(Dataset):

    def __init__(self, data_path, batch_size, n_way=10, k_shot=2, q_query=1):

        self.file_list = self.get_file_list(data_path)
        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

    def get_file_list(self, data_path):
        raise NotImplementedError('get_file_list function not implemented!')

    def get_one_task_data(self):
        raise NotImplementedError('get_one_task_data function not implemented!')

    def __len__(self):
        return len(self.file_list) // self.batch_size

    def __getitem__(self, index):
        return self.get_one_task_data()



