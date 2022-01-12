import torch

import os
import time
import itertools

from rich import print
from rich.tree import Tree as rTree

from ..data.dataset import Dataset
from ..data.samplers import Sampler, BatchSampler, ParallelSampler

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #  path = 'datasets/omniglot'
    path = 'datasets/dummy'

    dataset = Dataset(path, device)

    params = {
        'k': 5,
        'n': 5,
        'm': 1
    }

    def animate(tree, x):
        def callback(node, params):
            params['tree'] = params['tree'].add(node.key)
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
        os.system('clear')
        rtree = rTree('Dataset')
        tree.dfs(callback, {'tree': rtree})
        print(rtree)
        #  print(x)
        time.sleep(0.5)

    dataset.tree.put_samplers({
        #  2: (ParallelSampler, lambda x: x, {'batch_size': 2}),
        3: (BatchSampler, lambda x: x, {'batch_size': 2})
    })

    subtree = dataset.tree.get('train/class 1')
    visited = []
    for x in subtree:
        #  animate(subtree, list(itertools.chain(*x)))
        animate(subtree, x)
    animate(subtree, [])
