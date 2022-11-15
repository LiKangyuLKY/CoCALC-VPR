import datetime
import random
import os
from os.path import join

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from visualdl import LogWriter

from dataloader import DataCoCALC
from network import CoCALC
import warnings

warnings.filterwarnings('ignore')

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train(data_dir, save_dir, log_dir, epochs, batch_size, lr, lr_decay_step, show_step, save_iter):

    # 指定GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    # 初始化并行训练nccl后端
    dist.init_process_group(backend="nccl")
    # 配置当前进程使用哪块GPU
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if torch.cuda.device_count() > 1:
        print(">> Let's use", torch.cuda.device_count(), "GPUs!")

    model = CoCALC()
    train_data = DataCoCALC(data_dir)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_step, gamma=0.5)
    loss_fn = nn.MSELoss(reduction='mean')
    # 将模型移到对应的GPU
    model.to(device)

    # 模型并行化封装
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    print(">> Load the training dataset with a sample size of {}".format(
        train_data.__len__()))
    # 数据的并行化
    sampler = DistributedSampler(train_data)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size // torch.cuda.device_count(),
                              sampler=sampler,
                              pin_memory=True,
                              num_workers=8)

    # visualDL
    loss_wrt = LogWriter(logdir=join(
        log_dir, datetime.datetime.now().strftime('%b%d_%H-%M-%S')))

    iteration_num = 0
    lamb = 1
    print(">> Start training")
    for epoch in range(epochs):
        # 维持各个进程间相同的随机数种子
        sampler.set_epoch(epoch)

        model.train()
        start_time = datetime.datetime.now()
        loss_list = []
        for batch_id, (image, label) in enumerate(train_loader):
            iteration_num += 1
            image = image.to(device)
            label = label.to(device)

            out = model(image)  # 预测结果
            loss_all = loss_fn(out, label)

            optimizer.zero_grad()
            loss_all.backward()
            loss_list.append(loss_all.item())
            optimizer.step()

            if (batch_id+1) % show_step == 0 and local_rank == 0:
                msg = '%s | step: %d/%d/%d | loss_all=%.6f' % (
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_id+1, epoch+1, epochs, loss_all.item())
                print(msg)

            loss_wrt.add_scalar(
                tag='loss_all', step=iteration_num, value=loss_all.item())

            del image, label

        scheduler.step()    # 调整学习率

        end_time = datetime.datetime.now()
        spend = int((end_time-start_time).seconds)
        minu = spend // 60
        second = spend % 60

        loss_list = '%.5f' % np.mean(loss_list)
        if local_rank == 0:
            print('\nThis epoch spend {} m {} s, and the average loss is {}'.format(
                minu, second, loss_list))

        # 保存模型
        if (epoch+1) % save_iter == 0 and local_rank == 0:
            torch.save(model.module.state_dict(), join(save_dir, datetime.datetime.now(
            ).strftime('%b%d_%H-%M-%S')+'_'+str(epoch+1)+'.pkl'))
            print("Save model done.")


if __name__ == '__main__':
    data_dir = '../../dataset/slam/Pittsburgh250k/'  # place365_60000  Pittsburgh250k
    save_dir = 'checkpoint/'
    log_dir = './log/'
    epochs = 200
    batch_size = 512
    lr = 0.0001
    lr_decay_step = epochs / 2
    show_step = 30
    save_iter = epochs / 4
    # torchrun --nproc_per_node=8 train.py
    train(data_dir, save_dir, log_dir, epochs, batch_size,
          lr, lr_decay_step, show_step, save_iter)
