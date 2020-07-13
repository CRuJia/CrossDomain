import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

import pandas as pd
import numpy as np
import random

identity = lambda x: x
DATA_DIR = "/home/lemon/Datasets"


class SimpleDataset(Dataset):
    def __init__(self, data_file, transform, target_transform):
        df = pd.read_csv(os.path.join(DATA_DIR, data_file))
        self.images = list(df['filename'])
        self.labels = list(df['label'])
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path = os.path.join(DATA_DIR, self.images[index])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.labels[index])
        return img, target

    def __len__(self):
        return len(self.images)


class SubDataset(Dataset):
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform = identity, min_size=20):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        if len(self.sub_meta) < min_size:
            idxs = [i % len(self.sub_meta) for i in range(min_size)]
            self.sub_meta = np.array(self.sub_meta[idxs]).tolist()

    def __getitem__(self, index):
        img_path = os.path.join(DATA_DIR, self.sub_meta[index])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class SetDataset(Dataset):
    def __init__(self, data_file, batch_size, transform):
        df = pd.read_csv(os.path.join(DATA_DIR, data_file))
        self.images = list(df['filename'])
        self.labels = list(df['label'])
        self.cl_list = np.unique(self.labels).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.images,self.labels):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size = batch_size,
                                      shuffle = True,
                                      num_workers = 0,
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform, min_size=batch_size)  #TODO
            self.sub_dataloader.append(DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, index):
        return next(iter(self.sub_dataloader[index]))

    def __len__(self):
        return len(self.cl_list)


class MultiSetDataset(Dataset):
    def __init__(self, data_files, batch_size, transform):
        self.cl_list = np.array([])
        self.sub_dataloader = []
        self.n_classes = []
        for data_file in data_files:
            df = pd.read_csv(os.path.join(DATA_DIR, data_file))
            images = list(df['filename'])
            labels = list(df['label'])
            #TODO 是否要添加下一行
            # labels = list(map(lambda x:x+len(self.cl_list),labels))#不同数据集之间的类别 标签偏移
            cl_list = np.unique(labels).tolist()
            self.cl_list = np.concatenate((self.cl_list, cl_list))

            sub_meta = {}
            for cl in cl_list:
                sub_meta[cl] = []
            for x,y in zip(images,labels):
                sub_meta[y].append(x)

            sub_data_loader_params = dict(batch_size = batch_size,
                                          shuffle=True,
                                          num_workers=0,
                                          pin_memory=False)
            for cl in cl_list:
                sub_dataset = SubDataset(sub_meta[cl],cl, transform=transform, min_size=batch_size)
                self.sub_dataloader.append(DataLoader(sub_dataset, **sub_data_loader_params))
            self.n_classes.append(len(cl_list))

    def __getitem__(self, index):
        return next(iter(self.sub_dataloader[index]))

    def __len__(self):
        return len(self.cl_list)

    def lens(self):
        return self.n_classes


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


class MultiEpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.n_domains = len(n_classes)

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        domain_list = [i%self.n_domains for i in range(self.n_episodes)]# 按顺序从不同的domain中抽取数据
        random.shuffle(domain_list)
        for i in range(self.n_episodes):
            domain_idx = domain_list[i]
            start_idx = sum(self.n_classes[:domain_idx])
            yield torch.randperm(self.n_classes[domain_idx])[:self.n_way] + start_idx


if __name__ == '__main__':
    data_file = "CUB_200_2011/all.csv"
    df = pd.read_csv(os.path.join(DATA_DIR, data_file))
    images = list(df['filename'])
    labels = list(df['label'])
    cl_list = np.unique(labels).tolist()
    batch_size = 32
    sub_meta = {}
    for cl in cl_list:
        sub_meta[cl] = []

    for x,y in zip(images,labels):
        sub_meta[y].append(x)

    sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers=0,
                                  pin_memory=False,
                                  drop_last=False)
    transform = transforms.Compose([
        transforms.Resize((84,84)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])

    # dataset = SubDataset(sub_meta[0],cl_list[0], transform=transform, min_size=batch_size)
    # dataloader = DataLoader(dataset, **sub_data_loader_params)



    dataset = SetDataset(data_file=data_file, batch_size=batch_size, transform=transform)
    sampler = EpisodicBatchSampler(len(dataset), 5, 100)

    data_loader_params = dict(batch_sampler = sampler, num_workers = 4)
    data_loader = DataLoader(dataset, **data_loader_params)

    # print(data_loader)
    for i,(x,y) in enumerate(data_loader):
        print("i:",i)
        # print("x:",x)
        print("y:", y)
        print(len(y))
