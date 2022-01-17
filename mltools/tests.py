
import time

from rich import print
from rich.progress import track

from .data.tree import Tree
from .data.samplers import Sampler, BatchSampler
from .data.dataset import Dataset
from .data.loaders import FewShotLoader
from .util import Timer

if __name__ == '__main__':

    #  path = 'datasets/miniimagenet'
    #  path = 'datasets/dummy'
    path = 'datasets/omniglot'

    dataset = Dataset(path)
    train = dataset.split('train')
    test = dataset.split('test')
    sampler = FewShotLoader(dataset, {'k': 5, 'n': 5, 'batch_size': 10})

    s, q = next(sampler)
    print(s.shape, q.shape)
