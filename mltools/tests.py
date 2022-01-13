
from .data.tree import Tree
from .data.samplers import Sampler, BatchSampler, ParallelSampler
from .util import Timer
#  from rich import print

if __name__ == '__main__':

    timer = Timer()

    def iterate(t):
        for x in t:
            print(x)
        print(len(t))

    tree = timer.time(lambda: Tree('datasets/omniglot'), 'Tree __init__()')
    subtree = timer.time(lambda: tree.get('train'), 'Tree get()')
    timer.time(lambda: str(subtree), 'TreeNode __str__()')
    timer.time(lambda: iterate(subtree), 'TreeNode __iter__()')

    print(timer)
