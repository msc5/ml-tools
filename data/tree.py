import os
import pprint
import collections
import time

from pathlib import Path

from util import Color, Timer
from .samplers import Sampler, BatchSampler, ParallelSampler


class TreeNode:

    def __init__(self, key, path=None):
        self.key = key
        self.path = path
        self.files = []
        self.info = collections.defaultdict(int)
        self.children = {}
        self.sampler = Sampler
        self.params = {}
        self.callback = lambda x: x

    def __len__(self):
        if self.sampler:
            return len(self.sampler)
        else:
            return len(self.children)

    def __iter__(self):
        return self.generate()

    def add_child(self, path):
        child = TreeNode(path.name, path)
        if child.key in self.children:
            raise KeyError
        self.info['n_children'] += 1
        self.children[child.key] = child
        return child

    def add_file(self, path):
        self.info['n_files'] += 1
        self.files.append(Path(path))

    def generate(self):
        #  print(self.key, self.callback)
        if self.files:
            data = [self.callback(f) for f in self.files]
        else:
            data = list(self.children.values())
        sampler = self.sampler(data, self.params)
        for sample in sampler:
            if isinstance(sample, TreeNode):
                yield from sample.generate()
            elif isinstance(sample, list):
                yield sample


class Tree:

    def __init__(self, directory):
        self.path = Path(directory)
        self.root = TreeNode(self.path.name, self.path)
        self.build_from_dir()

    def __str__(self):
        """
        Returns a string representation of the Tree in DFS order
        """

        def format(vals, n):
            data = ''.join([f'{v:<40}' for v in vals])
            return f'{"":>{n * 4}}' + data

        def callback(node, params):
            msg.append(format([
                Color.CYAN(node.key),
                Color.YELLOW(str(node.sampler)),
                Color.GREEN(str(node.callback))
            ], node.info['depth']))
        msg = []
        self.dfs(callback)
        return '\n'.join(msg)

    def __iter__(self):
        yield from self.root

    def put_samplers(self, samplers, callback):
        def put_sampler(node, params):
            depth = node.info['depth']
            node.callback = callback
            if depth in samplers:
                node.sampler, node.params = samplers[depth]
            else:
                node.sampler, node.params = Sampler, {}
        self.dfs(put_sampler)
        return self

    def bfs(self, callback=None, params={}):
        queue = [self.root]
        while queue:
            curr = queue.pop(0)
            if callback:
                callback(curr, params)
            children = curr.children.items()
            for key, child in sorted(children):
                queue.append(child)

    def dfs(self, callback=None, params={}):
        def traverse(node, params):
            if callback:
                callback(node, params)
            children = node.children.items()
            for key, child in sorted(children):
                traverse(child, params.copy())
        traverse(self.root, params)

    def build_from_dir(self):
        def traverse(parent, depth):
            size = 1
            N_children = N_files = 0
            for path in parent.path.iterdir():
                if path.is_dir():
                    child = parent.add_child(path)
                    s, c, f = traverse(child, depth + 1)
                    size = s if s > size else size
                    N_children += c
                    N_files += f
                    self.levels['nodes'][depth + 1] += 1
                else:
                    parent.add_file(path)
                    N_files += 1
                    self.levels['files'][depth + 1] += 1
            parent.info['size'] = size
            parent.info['depth'] = depth
            parent.info['N_children'] = N_children
            parent.info['N_files'] = N_files
            return size + 1, N_children + 1, N_files
        self.levels = {
            'nodes': collections.defaultdict(int),
            'files': collections.defaultdict(int)
        }
        traverse(self.root, 0)
        self.size = self.root.info['size'] + 1
        self.levels['nodes'][0] = 1
        self.levels = {
            k: [
                v[d] for d in range(self.size)
            ] for k, v in self.levels.items()
        }


if __name__ == '__main__':

    pp = pprint.PrettyPrinter()
    timer = Timer()

    #  directory = 'datasets/miniimagenet'
    #  directory = 'datasets/omniglot'
    directory = 'datasets/dummy'

    tree = timer.time(lambda: Tree(directory), 'Tree __init__()')

    def callback(file):
        if isinstance(file, Path):
            return str(file)
        return file
    samplers = {
        2: (ParallelSampler, {'batch_size': 2}),
        3: (BatchSampler, {'batch_size': 2}),
    }
    timer.time(lambda: tree.put_samplers(
        samplers, callback), 'Tree put_samplers()')

    tree_str = timer.time(lambda: str(tree), 'Tree __str__()')
    #  print(tree_str)

    iteration = timer.time(
        lambda: [x for x in tree],
        'Tree iteration'
    )

    pp.pprint(iteration)

    print(timer)
