import torch

from torchvision.transforms import ToTensor

from PIL import Image
from statistics import mean
from pathlib import Path
from json import load

import os
import time
import pprint
import collections

import util

from .tests import iterate_dataset
from .tree import Tree


class Dataset:

    def __init__(
            self,
            directory: str,
            device: torch.device
    ):
        self.device = device
        self.tree = Tree(directory)
        self.path = Path(directory)
        self.name = self.path.name
        self.transform = ToTensor()
        try:
            config = load(open('config.json'))['datasets'][self.name]
            self.structure = ['Root', *config['structure']]
            self.class_level = config['class_level'] + 1
        except:
            print('Error Opening datasets.json')
            raise

    def __str__(self):
        """
        Returns a string representation of the dataset
        """
        size = self.tree.root.info['size']
        return self.name + '\n' + util.tabulate(
            ('Depth', range(size + 1)),
            ('Name', self.structure),
            ('Folder Count', self.tree.levels['nodes']),
            ('File Count', self.tree.levels['files']),
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

    times = collections.defaultdict(float)

    def timer(callback, label):
        start = time.perf_counter()
        val = callback()
        stop = time.perf_counter()
        times[label] = stop - start
        return val

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #  path = 'datasets/omniglot'
    path = 'datasets/miniimagenet'
    #  path = 'datasets/dummy'

    dataset = timer(lambda: FewShotDataset(path, device), 'DATASET INIT')
    _ = timer(lambda: print(dataset), 'DATASET __STR__')

    pp.pprint(dict(times))

    #  start = time.perf_counter()
    #  dataset = FewShotDataset(path, device)
    #  stop = time.perf_counter()
    #  print('INITIALIZE TIME: ', stop - start)
    #  start = time.perf_counter()
    #  print(dataset)
    #  stop = time.perf_counter()
    #  print('__STR__ TIME: ', stop - start)

    #  params = {
    #      'batch_size': 20,
    #      'full_permute': True,
    #      'k': 1,
    #      'n': 0,
    #      'm': 1,
    #  }

    #  print('ITERATING: ', params)
    #  start = time.perf_counter()
    #  iterator = dataset.split(params, 'train')
    #  n_seen = 0
    #  for i, x in enumerate(iterator):
    #      print(f'{i:4}', x.shape)
    #      n_seen += functools.reduce(lambda a, b: a * b, x.shape[:3])
    #      pass
    #  N_images = len(dataset.tree.get('train').all_children())
    #  stop = time.perf_counter()
    #  print('ITERATION TIME: ', stop - start)
    #  print('TOTAL IMAGES: ', N_images)
    #  print('SEEN IMAGES: ', n_seen)
    #  print('PERCENT PERMUTED: ', 100 * (n_seen / N_images))
