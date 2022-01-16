import torch

import os
import time
import itertools

from rich import print
from rich.tree import Tree as rTree
from rich.console import Console

from mltools.data.dataset import Dataset
from mltools.data.fewshot import FewShotDataset
from mltools.data.samplers import Sampler, BatchSampler
from mltools.data.loaders import FewShotLoader

if __name__ == '__main__':

    #  path = 'datasets/omniglot'
    path = 'datasets/dummy'

    def animate(tree, x):
        def callback(node, params):
            if node.info['depth'] == class_level:
                key = '[orange1]' + node.key
            elif node.info['depth'] == class_level - 1:
                key = '[red]' + node.key
            else:
                key = node.key
            if not 'root' in params:
                params['root'] = params['tree'] = rTree(key)
            else:
                params['tree'] = params['tree'].add(key)
            if node.files:
                for f in node.files:
                    if f in x:
                        f_str = '[green]' + f'{str(f.name):15}{"<---"}'
                        visited.append(f)
                    else:
                        if f in visited:
                            f_str = '[cyan]' + str(f.name)
                        else:
                            f_str = '[gray]' + str(f.name)
                    params['tree'].add(f_str)
        P = {}
        tree.dfs(callback, P, order='pre')
        os.system('clear')
        print(P['root'])
        time.sleep(0.2)

    class_level = 3
    #  S = [
    #      {
    #          'sampler': BatchSampler,
    #          'sample': 'all',
    #          'batch_size': 2,
    #          'batch_mode': 'random'
    #      },
    #      {
    #          'sampler': BatchSampler,
    #          'batch_size': 2,
    #          'batch_mode': 'random'
    #      }
    #  ]
    #  dataset.put_samplers(S)
    #  subtree = dataset.split('test')
    dataset = Dataset(path)
    #  split = dataset.split('test')
    loader = FewShotLoader(dataset, {'k': 2})
    visited = []
    for x in loader:
        #  print(x)
        animate(dataset.tree, list(itertools.chain(*x)))
        #  animate(subtree, x)
    #  animate(subtree, [])
