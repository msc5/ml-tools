import torch

import random
import pprint
import math


class Sampler:

    def __init__(self, data: list, params: object = None):
        """
        Implements a Sampler.
        Parameters:
            data: A list
            params: a dictionary that specifies parameters of the Sampler

        """
        assert hasattr(data, '__getitem__')
        self.data = sorted(
            [iter(d) if isinstance(d, Sampler) else d for d in data])
        self.params = params
        if self.params:
            self.batch_size = params.get('batch_size', 1)
            self.batch_mode = params.get('batch_mode', None)
        self.n = len(data)

    def __iter__(self):
        self.data = [iter(d) if isinstance(d, Sampler)
                     else d for d in self.data]
        self.i = 0
        self.v = self.data[0]
        return self

    def step(self):
        i = self.i
        v = self.v
        self.i += 1
        if self.i > self.n:
            raise StopIteration
        if self.i < self.n:
            self.v = self.data[self.i]
        return v

    def __next__(self):
        if isinstance(self.v, Sampler):
            try:
                return next(self.v)
            except StopIteration:
                self.step()
                return next(self)
        else:
            return self.step()

    def __len__(self):
        length = 0
        for c in self.data:
            if isinstance(c, Sampler):
                length += len(c)
            else:
                length += 1
        return length

    def __lt__(self, other):
        return len(self) < len(other)


class BatchSampler(Sampler):

    def __init__(self, data, params):
        super().__init__(data, params)
        assert self.batch_size <= self.n
        assert self.batch_size != 0

    def __iter__(self):
        self.data = [iter(d) if isinstance(d, Sampler)
                     else d for d in self.data]
        self.i = 0
        self.v = self.data[0:self.batch_size]
        return self

    def step(self):
        i = self.i
        v = self.v
        self.i += self.batch_size
        if self.i > self.n:
            raise StopIteration
        if self.i + self.batch_size <= self.n:
            self.v = self.data[self.i:self.i + self.batch_size]
        return v

    def swap(self):
        self.i += 1
        if self.i + self.batch_size > self.n:
            raise StopIteration
        self.v = self.data[self.i:self.i + self.batch_size]

    def __next__(self):
        values = []
        step = False
        swap = 0
        for v in self.v:
            if isinstance(v, Sampler):
                try:
                    values.append(next(v))
                except StopIteration:
                    if self.batch_mode is None:
                        self.step()
                    elif self.batch_mode == 'parallel':
                        self.swap()
                    return next(self)
            else:
                values.append(v)
                step = True
                if self.batch_mode == 'parallel':
                    self.swap()
        if step and self.batch_mode is None:
            self.step()
        assert len(values) == self.batch_size
        return values

    def __len__(self):
        length = 0
        batches = [[
            len(s) if isinstance(s, Sampler) else 1
            for s in self.data[i:i + self.batch_size]
        ] for i in range(0, self.n - 1, self.batch_size)]
        return sum([min(b) for b in batches])


class ParallelSampler(Sampler):

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
