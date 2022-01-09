import torch

from torchvision.transforms import ToTensor

from tqdm import tqdm
from PIL import Image
from statistics import mean

import os
import json
import random
import math
import itertools
import functools
import time
import pprint

import util

from .tests import iterate_dataset
from .tree import Tree
from .samplers import BatchSampler, RandomSampler, RandomBatchSampler


class Dataset:

    def __init__(
            self,
            directory: str,
            device: torch.device
    ):
        self.directory = directory
        self.device = device
        self.name = directory.split(os.sep)[-1]
        self.tree = Tree(directory=directory)
        try:
            path = os.path.join('datasets', 'datasets.json')
            config = json.load(open(path))[self.name]
            self.structure = config['structure']
            self.class_level = config['class_level'] + 1
            self.example_level = self.tree.depth() - 1
            self.transform = ToTensor()
        except:
            print('Error Opening datasets.json')
            raise

    def __str__(self):
        """
        Returns a string representation of the dataset
        """
        def callback(tree, level):
            levels[level].append((tree.n, tree.n_children))
        structure = ['Root', *self.structure]
        depth = self.tree.depth()
        levels = [[] for _ in range(depth)]
        self.tree.bfs(callback)
        names, averages, counts, types = [], [], [], []
        for i, level in enumerate(levels):
            names.append(structure[i])
            size, children = zip(*level)
            averages.append(mean(children))
            counts.append(len(size))
            if i == self.class_level:
                types.append('Class')
            elif i == self.example_level:
                types.append('Example')
            else:
                types.append('')
        return util.tabulate(
            ('Depth', range(depth)),
            ('Name', names),
            ('Type', types),
            ('Count', counts),
            ('Average Children', averages)
        )

    def __iter__(self):
        """
        Returns an iterable over the files in the dataset
        """
        return self.tree.generator(self.load_image)

    def __len__(self):
        """
        Returns the length of the dataset's total iteration
        """
        pass

    def load_image(self, tree):
        """
        Takes a tree as input and returns the file associated with its path
        """
        image = Image.open(tree.val)
        return self.transform(image).to(device)

    def split(self, directory):
        assert directory is not None
        split = self.tree.get(directory)
        return split.generator(self.load_image)

    def cat(self, tensors):
        return torch.cat([x.unsqueeze(0) for x in tensors])

    def batch(self, data, batch_size):
        params = {
            'batch_size': batch_size,
            'keep_last': True,
        }
        for batch in BatchSampler(data, params):
            yield self.cat(batch)


class FewShotDataset (Dataset):

    def __init__(
            self,
            directory: str,
            device: torch.device
    ):
        super().__init__(directory, device)

    def collate_images(self, iterator):
        """
        Turns the set of images returned by tree iteration into
        a task by collating lists with pytorch
        """
        for task in iterator:
            yield torch.cat([
                torch.cat([
                    c.unsqueeze(0)
                    for c in classes
                ], dim=0).unsqueeze(0)
                for classes in task
            ], dim=0)

    def split(self, params, directory):
        """
        Returns an iterator over images in the dataset with the
        specified parameters
        """
        assert directory is not None
        assert params is not None
        full_permute = params.get('full_permute', False)
        batch_size = params.get('batch_size', 1)
        k = params['k']
        n = params['n']
        m = params['m']
        self.tree.put_iters([
            (self.class_level - 1, RandomBatchSampler, {
                'batch_size': k,
                'full_permute': full_permute
            }),
            (self.class_level, RandomBatchSampler, {
                'batch_size': n + m,
                'full_permute': full_permute
            })
        ])
        split = self.tree.get(directory)
        images = split.generator(self.load_image)
        tasks = self.collate_images(images)
        batches = self.batch(tasks, batch_size)
        yield from batches


if __name__ == '__main__':

    pp = pprint.PrettyPrinter()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #  path = 'datasets/omniglot'
    #  path = 'datasets/miniimagenet'
    path = 'datasets/dummy'

    start = time.perf_counter()
    dataset = FewShotDataset(path, device)
    stop = time.perf_counter()
    print('INITIALIZE TIME: ', stop - start)
    print(dataset)

    params = {
        'batch_size': 20,
        'full_permute': True,
        'k': 1,
        'n': 0,
        'm': 1,
    }

    print('ITERATING: ', params)
    start = time.perf_counter()
    iterator = dataset.split(params, 'train')
    n_seen = 0
    for i, x in enumerate(iterator):
        print(f'{i:4}', x.shape)
        n_seen += functools.reduce(lambda a, b: a * b, x.shape[:3])
        pass
    N_images = len(dataset.tree.get('train').all_children())
    stop = time.perf_counter()
    print('ITERATION TIME: ', stop - start)
    print('TOTAL IMAGES: ', N_images)
    print('SEEN IMAGES: ', n_seen)
    print('PERCENT PERMUTED: ', 100 * (n_seen / N_images))
