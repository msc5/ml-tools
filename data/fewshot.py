import torch

from torchvision.transforms import ToTensor

from PIL import Image
from statistics import mean

import os
import json
import random
import math
import itertools
import time
import pprint

import util

from .tests import iterate_dataset
from .tree import Tree
from .samplers import BatchSampler, RandomSampler, RandomBatchSampler


class Dataset:

    def __init__(
            self,
            directory: str
    ):
        self.directory = directory
        self.name = directory.split(os.sep)[-1]
        self.tree = Tree(directory=directory)
        try:
            path = os.path.join('datasets', 'datasets.json')
            config = json.load(open(path))[self.name]
            self.structure = config['structure']
            self.class_level = config['class_level'] + 1
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

        names = []
        averages = []
        counts = []
        for i, level in enumerate(levels):
            names.append(structure[i])
            size, children = zip(*level)
            averages.append(mean(children))
            counts.append(len(size))

        return util.tabulate(
            ('Depth', range(depth)),
            ('Name', names),
            ('Count', counts),
            ('Average Children', averages)
        )

    def __iter__(self):
        """
        Returns an iterable over the files in the dataset
        """
        return self.tree.generator(self.load_image)

    def load_image(self, tree):
        """
        Takes a tree as input and returns the file associated with its path
        """
        image = Image.open(tree.val)
        return self.transform(image)

    def split(self, directory):
        assert directory is not None
        split = self.tree.get(directory)
        return split.generator(self.load_image)

#      def __get_images(self, keys):
#          # image = Image.open(os.path.join(self.dir, p))
#          images = [
#              torch.cat([
#                  self.transform(
#                      Image.open(
#                          os.path.join(self.dir, p)
#                      )).unsqueeze(0) for p in k
#              ]).unsqueeze(0) for k in keys
#          ]
#          return torch.cat(images, dim=0).to(self.device)
#
#      def __getitem__(self, keys):
#          s, q = keys
#          support = self.__get_images(s)
#          query = self.__get_images(q)
#          return support, query


class FewShotDataset (Dataset):

    def __init__(
            self,
            directory: str
    ):
        super().__init__(directory)

    def get_iter(self, split, params):
        """
        Returns an iterator over images in the dataset with the
        specified parameters
        """
        bs = params['batch_size']
        k = params['k']
        n = params['n']
        m = params['m']
        self.tree.put_iters([
            (self.class_level - 1, RandomBatchSampler, {'batch_size': k}),
            (self.class_level, RandomBatchSampler, {'batch_size': n + m})
        ])
        return self.split(split)


if __name__ == '__main__':

    pp = pprint.PrettyPrinter()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #  path = 'datasets/omniglot'
    path = 'datasets/miniimagenet'
    #  path = 'datasets/dummy'

    bs = 20
    k = 5
    n = 5
    m = 1

    start = time.perf_counter()
    dataset = FewShotDataset(path)
    stop = time.perf_counter()
    print(stop - start)
    print(dataset)
    #  print(len(dataset))

    params = {
        'batch_size': 20,
        'k': 5,
        'n': 5,
        'm': 1,
    }

    start = time.perf_counter()
    for i, x in enumerate(dataset.split('train')):
        pass
        print(i)
        pp.pprint(x.shape)
    stop = time.perf_counter()
    print(stop - start)
