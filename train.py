# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2022/7/6 10:01
# @Software: PyCharm
# @Brief:

import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from core.helper import get_model, get_dataset
from net.maml import maml_train
from args import args, dev


if __name__ == '__main__':
    model = get_model(args, dev)
    train_dataset, val_dataset = get_dataset(args)

    train_loader = DataLoader(train_dataset, batch_size=args.task_num, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.val_task_num, shuffle=False, num_workers=args.num_workers)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, args.outer_lr)
    best_acc = 0

    model.train()
    for epoch in range(args.epochs):
        train_acc = []
        val_acc = []
        train_loss = []
        val_loss = []

        train_bar = tqdm(train_loader)
        for support_images, support_labels, query_images, query_labels in train_bar:
            train_bar.set_description("epoch {}".format(epoch + 1))
            # Get variables
            support_images = support_images.float().to(dev)
            support_labels = support_labels.long().to(dev)
            query_images = query_images.float().to(dev)
            query_labels = query_labels.long().to(dev)

            loss, acc = maml_train(model, support_images, support_labels, query_images, query_labels,
                                   1, args, optimizer)

            train_loss.append(loss.item())
            train_acc.append(acc)
            train_bar.set_postfix(loss="{:.4f}".format(loss.item()))

        for support_images, support_labels, query_images, query_labels in val_loader:

            # Get variables
            support_images = support_images.float().to(dev)
            support_labels = support_labels.long().to(dev)
            query_images = query_images.float().to(dev)
            query_labels = query_labels.long().to(dev)

            loss, acc = maml_train(model, support_images, support_labels, query_images, query_labels,
                                   3, args, optimizer, is_train=False)

            # Must use .item()  to add total loss, or will occur GPU memory leak.
            # Because dynamic graph is created during forward, collect in backward.
            val_loss.append(loss.item())
            val_acc.append(acc)

        print("=> loss: {:.4f}   acc: {:.4f}   val_loss: {:.4f}   val_acc: {:.4f}".
              format(np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc)))

        if np.mean(val_acc) > best_acc:
            best_acc = np.mean(val_acc)
            torch.save(model, 'best.pt')
