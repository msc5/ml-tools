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
from .samplers import Sampler, BatchSampler, ParallelSampler


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
        pass

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
            'batch_mode': 'keep_last',
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

    def get_iter(self, params):
        """
        Returns an iterator over images in the dataset with the
        specified parameters
        """
        def load_image(path): return self.transform(Image.open(path))
        batch_mode = params.get('batch_mode', 'normal')
        batch_size = params.get('batch_size', 1)
        k, n, m = params.get('k', 1), params.get('n', 1), params.get('m', 1)
        self.tree.put_samplers({
            self.class_level - 1: (ParallelSampler, {'batch_size': k}),
            self.class_level: (BatchSampler, {'batch_size': n + m})
        }, load_image)
        #  collated = self.collate_images(self.tree)
        #  batched = self.batch(collated, batch_size)
        #  yield from batched


if __name__ == '__main__':

    pp = pprint.PrettyPrinter()
    timer = util.Timer()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #  path = 'datasets/omniglot'
    #  path = 'datasets/miniimagenet'
    path = 'datasets/dummy'

    params = {
        'batch_size': 10,
        'k': 2,
        'n': 1,
        'm': 1
    }

    dataset = timer.time(
        lambda: FewShotDataset(path, device),
        'dataset __init__()'
    )
    timer.time(lambda: print(dataset), 'dataset __str__()')
    iteration = timer.time(
        lambda: [x for x in dataset.get_iter(params)], 'dataset __iter__()')
    #  timer.time(lambda: dataset.split(params, 'train'), 'dataset split()')

    pp.pprint(iteration)

    print(timer)
