# -*- coding: utf-8 -*-
# @File : __init__.py
# @Author: Runist
# @Time : 2022/7/6 10:50
# @Software: PyCharm
# @Brief:
from .dataset import MAMLDataset
from .omniglot import OmniglotDataset

__all__ = ['MAMLDataset', 'OmniglotDataset']