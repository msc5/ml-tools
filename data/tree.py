import os
import pprint
import collections
import time

from pathlib import Path

from .samplers import Sampler, BatchSampler, ParallelSampler


class TreeNode:

    def __init__(self, key, path=None):
        self.key = key
        self.path = path
        self.files = []
        self.info = collections.defaultdict(int)
        self.children = collections.OrderedDict()
        self.sampler = None

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

    def __iter__(self):
        if self.files:
            yield from self.files
        elif not self.sampler:
            children = list(self.children.values())
            self.sampler = Sampler(children, {'batch_size': 2})
            for sample in self.sampler:
                if isinstance(sample, list):
                    yield sample
                else:
                    yield from sample
        elif self.sampler:
            yield from self.sampler


class Tree:

    def __init__(self, directory):
        self.path = Path(directory)
        self.root = TreeNode(self.path.name, self.path)
        self.build_from_dir()

    def __str__(self):
        """
        Returns a string representation of the Tree in DFS order
        """

        def space(v, n): return f'{"":>{n * 5}}{v}'

        def callback(node, params):
            msg.append(space(node.key, params['depth']))
            params['depth'] += 1

        msg = []
        self.dfs(callback, {'depth': 0})
        return '\n'.join(msg)

    def __iter__(self):
        yield from self.root

    def bfs(self, callback=None, params={}):
        queue = [self.root]
        while queue:
            curr = queue.pop(0)
            if callback:
                callback(curr, params)
            for key, child in curr.children.items():
                queue.append(child)

    def dfs(self, callback=None, params={}):
        def traverse(node, params):
            if callback:
                callback(node, params)
            for key, child in node.children.items():
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
        self.levels = {
            k: [
                v[d] for d in range(self.size)
            ] for k, v in self.levels.items()
        }


if __name__ == '__main__':

    pp = pprint.PrettyPrinter()

    #  directory = 'datasets/miniimagenet'
    #  directory = 'datasets/omniglot'
    directory = 'datasets/dummy'

    start = time.perf_counter()
    tree = Tree(directory)
    stop = time.perf_counter()
    print(f'{"Tree Initialization":>30} :', stop - start, 'ms')
    #  print(tree)

    for i, x in enumerate(tree):
        print(i)
        pp.pprint(x)
