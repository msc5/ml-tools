import random
import pprint
import math


class Sampler:

    def __init__(self, children, params):
        self.children = children
        self.batch_size = params.get('batch_size', 1)
        self.batch_mode = params.get('batch_mode', None)
        self.n = len(children)
        self.i = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        idx = self.i
        self.i += 1
        if self.i > self.n:
            raise StopIteration
        return self.children[idx]


class BatchSampler (Sampler):

    def __init__(self, children, params):
        super().__init__(children, params)
        self.i = 0

    def __len__(self):
        return self.n // self.batch_size

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        start = self.i
        self.i += self.batch_size
        if self.i > self.n:
            raise StopIteration
        batch = self.children[start:self.i]
        return batch


class ParallelSampler (Sampler):

    def __init__(self, children, params):
        super().__init__(children, params)
        self.children = children
        self.parallel = [iter(c) for c in children]
        self.iterators = self.parallel[:self.batch_size]
        self.j = self.batch_size
        self.i = 0

    def __len__(self):
        lengths = [0 for _ in range(self.batch_size)]
        for i, c in enumerate(self.children):
            lengths[i % self.batch_size] += len(c)
        return min(lengths)

    def __iter__(self):
        self.parallel = [iter(c) for c in self.children]
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
