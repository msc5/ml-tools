import os
import pprint
import random
import itertools


class Tree:

    def __init__(
            self,
            path: str = None,
            val: object = None,
            directory: str = None
    ):
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        self.children = {}
        self.iter = None
        self.next_trees = []
        if self.directory:
            self.put_path(self.directory)

    def __str__(self):
        info = []

        def f(v, n): return f'{str(v):{n}}' if v else ''

        def cb(tree, level):
            info.append(
                str(level) + ' : ' +
                f(tree.path, 15) +
                f(tree.val, 50) +
                f(tree.iter, 40)
            )

        self.bfs(cb)
        return '\n'.join(info) + '\n'

    def __iter__(self):
        if self.iter:
            for iteration in self.iter:
                if isinstance(iteration, Tree):
                    yield from iteration
                if isinstance(iteration, list):
                    yield from zip(*iteration)
        else:
            yield self.val

    def put_path(self, path):
        for root, dirs, files in os.walk(path):
            self.put(root, root)
            if files:
                for f in files:
                    p = os.path.join(root, f)
                    self.put(p, p)

    def put_iters(self, iters):
        assert iters is not None
        assert len(iters) == self.depth() - 1

        def cb(tree, level):
            children = list(tree.get_children())
            if children:
                gen, params = iters[level]
                tree.iter = gen(children, params)
        self.bfs(cb)

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
            return new_tree
        elif not key in self.children:
            new_tree = Tree(key)
            self.children[key] = new_tree
            return new_tree.put(next_path, val)
        elif key in self.children:
            curr = self.children[key]
            return curr.put(next_path, val)

    def get(self, path):
        assert path is not None
        paths = path.split(os.sep)
        next_path = os.sep.join(paths[1:])
        key = paths[0]
        if key == self.path:
            return self.get(next_path)
        if not key in self.children:
            return None
        if key in self.children:
            if len(paths) == 1:
                return self.children[key]
            return self.children[key].get(next_path)

    def get_children(self, key=None):
        if not key:
            root = self
        else:
            root = self.get(key)
        assert root is not None
        return root.children.values()

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


def rs(data, params):
    s = data.copy()
    random.shuffle(s)
    yield from s


def bs(data, params):
    batch_size = params['batch_size']
    keep_last = params['keep_last']
    try:
        while True:
            val = []
            for i in range(batch_size):
                val.append(next(data))
            yield val
    except StopIteration:
        if keep_last:
            yield val
    finally:
        del data


def rbs(data, params):
    batch_size = params['batch_size']
    for x in bs(rs(data, params), params):
        yield x


def prbs(data, params):
    parallel_size = params['parallel_size']
    parallels = [rbs(data, params) for _ in range(parallel_size)]
    try:
        while True:
            yield [next(g) for g in parallels]
    except StopIteration:
        pass
    finally:
        del data


if __name__ == '__main__':

    pp = pprint.PrettyPrinter(indent=4)

    #  chars = 'abcdefghijklmnopqrstuvwxyz'
    #  data = [''.join(random.sample(chars, 5)) for _ in range(20)]
    #  pp.pprint(data)
    #  gen = rbs(data, {'batch_size': 7})
    #  for d in gen:
    #      pp.pprint(d)

    tree = Tree()

    #  path = 'datasets/omniglot'
    path = 'datasets/miniimagenet'
    tree.put_path(path)
    print('Depth: ', tree.depth())

    k = 20
    n = 5
    m = 1
    tree.put_iters([
        (rs, {}),
        (rs, {}),
        (rbs, {'batch_size': k, 'keep_last': False}),
        (rbs, {'batch_size': n + m, 'keep_last': False}),
    ])

    N_images = len(tree.all_children())
    print('Total Files: ', N_images)

    for i, x in enumerate(tree):
        print(i)
        pp.pprint(x)
        pass
    n_images = i * k * (n + m)
    print(n_images)
    print(n_images / N_images)

    #  print('Classes:')
    #  classes = tree.get_level(level=3)
    #  pp.pprint([c.val for c in classes])
    #
    #  print('Data:')
    #  children = tree.all_children()
    #  pp.pprint([c.val for c in children])
