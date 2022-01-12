import torch

import random
import pprint
import math


class Sampler:

    def __init__(self, data, params):
        """
        Implements a Sampler.
        Parameters:
            data:   an object that implements '__len__()' and either
                    '__getitem__()' or '__next__()'.
                    (e.g., a Sampler, a list, or a TreeNode)
            params: a dictionary that specifies parameters of the Sampler

        """
        if hasattr(data, '__getitem__'):
            self.data = data
            self.type = 'map'
        elif hasattr(data, '__iter__'):
            self.iterable = data
            self.data = iter(data)
            self.type = 'iter'
        else:
            raise ValueError
        assert params is not None
        self.batch_size = params.get('batch_size', 1)
        self.batch_mode = params.get('batch_mode', None)
        self.n = len(data)
        self.i = 0

    def __len__(self):
        length = 0
        for c in self.data:
            if hasattr(c, '__len__'):
                length += len(c)
            else:
                length += 1
        return length

    def __iter__(self):
        if self.type == 'iter':
            self.data == iter(self.iterable)
            return self
        if self.type == 'map':
            self.i = 0
            return self

    def __next__(self):
        if self.type == 'iter':
            return next(self.data)
        if self.type == 'map':
            i = self.i
            self.i += 1
            if self.i > self.n:
                raise StopIteration
            return self.data[i]


class BatchSampler (Sampler):

    def __init__(self, data, params):
        super().__init__(data, params)

    def __len__(self):
        return self.n // self.batch_size

    def __iter__(self):
        if self.type == 'iter':
            self.data == iter(self.iterable)
            return self
        if self.type == 'map':
            self.i = 0
            return self

    def __next__(self):
        if self.type == 'iter':
            #  vals = []
            #  for _ in range(self.batch_size):
            #      vals.append(next(self.data))
            #  return vals
            return [next(self.data) for _ in range(self.batch_size)]
        if self.type == 'map':
            start = self.i
            self.i += self.batch_size
            if self.i > self.n:
                raise StopIteration
            batch = self.data[start:self.i]
            return batch


class ParallelSampler (Sampler):

    def __init__(self, data, params):
        super().__init__(data, params)
        self.data = sorted(data)
        self.parallel = [iter(c) for c in data]
        self.iterators = self.parallel[:self.batch_size]
        self.j = self.batch_size
        self.i = 0

    def __len__(self):
        lengths = [0 for _ in range(self.batch_size)]
        for i, c in enumerate(self.data):
            lengths[i % self.batch_size] += len(c)
        return min(lengths)

    def __iter__(self):
        self.parallel = [iter(c) for c in self.data]
        self.iterators = self.parallel[:self.batch_size]
        self.j = self.batch_size
        self.i = 0
        return self

    def __next__(self):
        vals = []
        for i, iterator in enumerate(self.iterators):
            try:
                vals.append(next(iterator))
            except StopIteration:
                if self.j >= self.n:
                    raise StopIteration
                self.iterators[i] = self.parallel[self.j]
                self.j += 1
                vals.append(next(self.iterators[i]))
        return vals


class CollateSampler (Sampler):

    def __init__(self, data, params):
        super().__init__(data, params)

    def __iter__(self):
        if self.type == 'iter':
            self.data == iter(self.iterable)
            return self
        if self.type == 'map':
            self.i = 0
            return self

    def __next__(self):
        print('COLLATENEXT')
        if self.type == 'iter':
            print(self.iterable)
            print('nextdata', self.data)
            data = next(self.data)
            print('nextdata')
            return self.dfs(data)
        if self.type == 'map':
            raise ValueError

    def dfs(self, parent):
        for i, child in enumerate(parent):
            print('child')
            if isinstance(child, list):
                parent[i] = self.dfs(child)
            else:
                child.unsqueeze(0)
        return torch.cat(parent)


if __name__ == '__main__':

    pp = pprint.PrettyPrinter(width=300)

    #  data = [f'{i:3}' for i in range(30)]
    #  #  sampler = BatchSampler(data, {'batch_size': 3})
    #  sampler = Sampler(data, {})
    #  print(data)
    #  for i, x in enumerate(sampler):
    #      print(i, x)

    m = (5, 20)
    data = [
        [
            f'{j - m[0] + 1:3.0f}_{i:<3.0f}' for i in range(j)
        ] for j in range(*m)
    ]
    pp.pprint([[a for a in b] for b in data])
    sampler = ParallelSampler(
        data,
        {'batch_size': m[0]}
    )
    print(len(sampler))
    #  sampler = ParallelSampler(iter(iter(d)
    #                            for d in data), {'batch_size': m[0]})

    for i, x in enumerate(sampler):
        print(f'{i:3}', x)
    n_data = (m[1] - m[0]) * (m[0] + (m[1] - m[0] - 1) // 2)
    n_seen = (i + 1) * m[0]
    n_norm = m[0]**2
    print('# Data: ', n_data)
    print('# Seen: ', n_seen)
    print('# Norm: ', n_norm)
    print('Permuted: ', n_seen / n_data, '%')
