import torch

from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler, RandomSampler
from torchvision.transforms import ToTensor

from PIL import Image

import os
import json
import random
import math
import itertools
import time

import util

from .tests import iterate_dataset
from .tree import Tree
from .iterators import NestedIterator, ParallelIterator


class FewShotDataset(Dataset):

    def __init__(
            self,
            path,
            split='train',
            structure=['Class'],
            class_level=0,
            device='cpu',
    ):
        self.dir = os.path.join('datasets', path)
        self.split = split
        self.structure = structure
        self.class_level = class_level
        self.device = device
        self.transform = ToTensor()
        self.__name__ = 'FewShotDataset [' + path + ']'
        self.tree = Tree()
        for root, dirs, files in os.walk(self.dir):
            self.tree.put(root, root)
            for f in files:
                p = os.path.join(root, f)
                self.tree.put(p, p)

    def __str__(self):
        """
        Returns a string representation of the dataset
        """
        classes = self.classes()
        num_classes = len(classes)
        num_examples = 0
        for c in classes:
            num_examples += len(self.examples(c))
        avg_examples = int(num_examples / num_classes)
        msg = (
            f'{"Number of Classes: ":>40}{num_classes}\n'
            f'{"Number of Examples: ":>40}{num_examples}\n'
            f'{"Average Number of Examples per Class: ":>40}{avg_examples}'
        )
        return msg

    def classes(self):
        """
        Returns a list of all classes in the dataset
        """
        return self.tree.get_level(level=self.class_level)

    def examples(self, key=None):
        """
        Returns a list of all examples in the dataset for a given class
        """
        return self.tree.get_children(key)

    def __get_images(self, keys):
        # image = Image.open(os.path.join(self.dir, p))
        images = [
            torch.cat([
                self.transform(
                    Image.open(
                        os.path.join(self.dir, p)
                    )).unsqueeze(0) for p in k
            ]).unsqueeze(0) for k in keys
        ]
        return torch.cat(images, dim=0).to(self.device)

    def __getitem__(self, keys):
        s, q = keys
        support = self.__get_images(s)
        query = self.__get_images(q)
        return support, query


class FewShotSampler(Sampler):

    def __init__(self, dataset, k=5, n=1, m=1):
        assert dataset is not None
        self.ds = dataset
        self.k = k
        self.n = n
        self.m = m
        self.K = len(dataset.classes())
        self.N = int(len(dataset.examples()) / self.K)
        cs = BatchSampler(
            RandomSampler(range(self.K)), k, True)
        es = BatchSampler(
            ParallelIterator(
                RandomSampler(range(self.N)), k), n + m, True)
        self.gen = NestedIterator(cs, es)
        self.iter = iter(self.gen)

    def __iter__(self):
        self.iter = iter(self.gen)
        return self

    def __next__(self):
        keys = next(self.iter)
        return paths

    def __len__(self):
        return len(self.gen)


def emitFewShotLoader(data, device, split, bs, k, n, m):
    try:
        path = os.path.join('datasets', 'datasets.json')
        config = json.load(open(path))[data]
        structure = config['structure']
        class_level = config['class_level']
    except:
        print('Error Opening datasets.json')
    ds = FewShotDataset(
        data,
        split=split,
        structure=structure,
        class_level=class_level + 3,
        device=device,
    )
    s = FewShotSampler(ds, k=k, n=n, m=m)
    dl = DataLoader(ds, sampler=s, batch_size=bs, drop_last=True)
    return dl


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data = 'omniglot'
    bs = 20
    k = 5
    n = 5
    m = 1

    dl = emitFewShotLoader(data, device, 'train', bs, k, n, m)
    ds = dl.dataset

    iterate_dataset(dl, ds)
