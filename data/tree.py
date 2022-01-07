import os
import time
import pprint
import random
import itertools

from .samplers import SimpleSampler, BatchSampler, RandomBatchSampler


class Tree:

    def __init__(
            self,
            path: str = None,
            val: object = None,
            directory: str = None,
            iters: list = None
    ):
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        self.children = {}
        self.iter = None
        self.n = 0
        self.n_children = 0
        self.n_iter = 0
        if self.directory:
            self.put_directory(self.directory)
        if self.iters:
            self.put_iters(self.iters)

    def __str__(self):
        info = []

        def f(v, n): return f'{str(v):{n}}' if v is not None else f'{"":{n}}'

        def cb(tree, level):
            info.append(
                str(level) + ' : ' +
                f(tree.path, 15) +
                f(tree.val, 60) +
                f(tree.iter, 60) +
                f(tree.n, 10) +
                f(tree.n_children, 10) +
                f(tree.n_iter, 10)
            )

        self.bfs(cb)
        return '\n'.join(info) + '\n'

    def __iter__(self):
        def callback(tree): yield self.val
        return self.generator(callback)

    def generator(self, callback):
        assert callback is not None
        if self.iter:
            for iteration in self.iter:
                if isinstance(iteration, Tree):
                    #  yield from iteration
                    yield from iteration.generator(callback)
                if isinstance(iteration, list):
                    #  yield from map(list, zip(*iteration))
                    yield from map(list, zip(*[i.generator(callback) for i in iteration]))
        else:
            yield callback(self)

    def put_directory(self, directory):
        self.directory = directory
        cut = len(os.path.split(self.directory)) - 1
        for root, dirs, files in os.walk(self.directory):
            rel_root = os.path.join(*root.split(os.sep)[cut:])
            self.put(rel_root, root)
            if files:
                for f in files:
                    p = os.path.join(rel_root, f)
                    v = os.path.join(root, f)
                    self.put(p, v)
        for level in range(self.depth() - 1):
            self.put_iters([(level, SimpleSampler, {})])

    def put_iters(self, iters):
        assert iters is not None
        for item in iters:
            self.put_iter(item)

    def put_iter(self, item):
        assert item is not None
        level, generator, params = item
        for tree in self.get_level(level=level):
            tree.iter = generator(tree.get_children(), params)
            batch_size = params.get('batch_size', 1)
            tree.n_iter = tree.n_children // batch_size

    def depth(self):
        max_depth = [0]

        def cb(tree, level):
            if level > max_depth[0]:
                max_depth[0] = level
        self.bfs(cb)
        return max_depth[0] + 1

    def bfs(self, cb=None):
        q = [(self, 0)]
        while q:
            curr, level = q.pop(0)
            if cb:
                cb(curr, level)
            for key, tree in curr.children.items():
                q.append((tree, level + 1))

    def dfs(self, cb=None):
        if cb:
            cb(self)
        for tree in self.get_children():
            tree.dfs(cb)

    def put(self, path, val=None):
        assert path is not None
        self.n += 1
        paths = path.split(os.sep)
        next_path = os.sep.join(paths[1:])
        key = paths[0]
        if not self.path:
            self.path = key
        if key == self.path:
            if len(paths) == 1:
                self.key = key
                self.val = val
                return self
            return self.put(next_path, val)
        elif len(paths) == 1:
            new_tree = Tree(path, val)
            self.children[path] = new_tree
            self.n_children += 1
            return new_tree
        elif not key in self.children:
            new_tree = Tree(key)
            self.children[key] = new_tree
            self.n_children += 1
            return new_tree.put(next_path, val)
        elif key in self.children:
            curr = self.children[key]
            return curr.put(next_path, val)

    def get(self, path):
        assert path is not None
        paths = path.split(os.sep)
        next_path = os.sep.join(paths[1:])
        key = paths[0]
        #  print('PATH: ', self.path, key)
        if key == self.path:
            return self.get(next_path)
        if key in self.children:
            if len(paths) == 1:
                return self.children[key]
            return self.children[key].get(next_path)
        else:
            return None

    def get_children(self, key=None):
        if not key:
            root = self
        else:
            root = self.get(key)
        assert root is not None
        return list(root.children.values())

    def all_children(self, key=None):
        if not key:
            root = self
        else:
            root = self.get(key)
        assert root is not None
        children = []

        def collect(tree, level):
            if not tree.children:
                children.append(tree)

        self.bfs(collect)
        return children

    def get_level(self, key=None, level=0):
        if not key:
            root = self
        else:
            root = self.get(key)
        assert root is not None
        trees = []

        def collect(tree, l):
            if l == level:
                trees.append(tree)

        self.bfs(collect)
        return trees

    def all_levels(self, key=None):
        if not key:
            root = self
        else:
            root = self.get(key)
        assert root is not None
        return [self.get_level(level=l) for l in range(self.depth())]


if __name__ == '__main__':

    pp = pprint.PrettyPrinter(indent=4)

    #  chars = 'abcdefghijklmnopqrstuvwxyz'
    #  data = [''.join(random.sample(chars, 5)) for _ in range(50)]
    #  pp.pprint(data)
    #  gen = rbs(data, {'batch_size': 7, 'keep_last': False})
    #  for d in gen:
    #      pp.pprint(d)

    #  path = 'datasets/omniglot'
    #  path = 'datasets/miniimagenet'
    path = 'datasets/dummy'

    start = time.perf_counter()
    tree = Tree(directory=path)
    stop = time.perf_counter()
    print(stop - start)
    #  print(tree)

    #  target = tree.get('split 1')
    #  print(tree)
    #  print(target)

    print('Depth: ', tree.depth())

    k = 1
    n = 0
    m = 1

    tree.put_iters([
        (2, RandomBatchSampler, {'batch_size': k}),
        (3, RandomBatchSampler, {'batch_size': n + m}),
    ])

    print(tree)

    N_images = len(tree.all_children())
    print('Total Files: ', N_images)
    def callback(tree): return 'funny' + tree.val
    for i, x in enumerate(tree.generator(callback)):
        print(i)
        pp.pprint(x)
        pass
    n_images = (i + 1) * k * (n + m)
    print(n_images)
    print(n_images / N_images)

    #  print('Classes:')
    #  classes = tree.get_level(level=3)
    #  pp.pprint([c.val for c in classes])
    #
    #  print('Data:')
    #  children = tree.all_children()
    #  pp.pprint([c.val for c in children])
