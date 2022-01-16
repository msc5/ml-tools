import torch

import random
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
        self.data = data
        self.params = params
        if params is not None:
            self.batch_size = params.get('batch_size', 1)
            self.batch_mode = params.get('batch_mode', 'wrap')
            self.batch_order = params.get('batch_order', 'random')
            self.callback = params.get('callback', None)
        else:
            self.callback = lambda x: x
        self.n = len(data)

    def __iter__(self):
        sampler = Sampler(self.data, self.params)
        sampler.data = [iter(d) if isinstance(d, Sampler)
                        else d for d in self.data]
        sampler.i = 0
        sampler.v = sampler.data[0]
        return sampler

    def step(self):
        i = self.i
        v = self.v
        self.i += 1
        if self.i > self.n:
            raise StopIteration
        if self.i < self.n:
            self.v = self.data[self.i]
        return self.callback(v) if self.callback is not None else v

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
        assert self.batch_mode in ['cutoff', 'wrap']
        assert self.batch_order in ['sorted', 'random']

    def __iter__(self):
        sampler = BatchSampler(self.data, self.params)

        if self.batch_mode == 'sorted':
            sampler.data = sorted(sampler.data)
        if self.batch_mode == 'random':
            random.shuffle(sampler.data)

        def gd(a, b): return a // b + (1 if a % b != 0 else 0)
        sampler.i = 0
        sampler.l = gd(sampler.n, sampler.batch_size)
        print(sampler, sampler.l)
        sampler.p = list(range(sampler.batch_size))
        sampler.v = [sampler.data[p] for p in sampler.p]
        sampler.v = [iter(v) if isinstance(v, Sampler)
                     else v for v in sampler.v]
        return sampler

    def step(self):
        self.i += 1
        if self.i >= self.l:
            raise StopIteration
        self.p = [(p + self.batch_size) % self.n for p in self.p]
        self.v = [self.data[p] for p in self.p]
        self.v = [iter(v) if isinstance(v, Sampler)
                  else v for v in self.v]

    def __next__(self):
        values = []
        step = False
        for v in self.v:
            if isinstance(v, Sampler):
                try:
                    values.append(next(v))
                except StopIteration:
                    self.step()
                    return next(self)
            else:
                values.append(v)
                step = True
        if step:
            self.step()
        assert len(values) == self.batch_size
        return self.callback(values) if self.callback is not None else values

    def __len__(self):
        pass
        #  def gd(a, b): return a // b + (1 if a % b != 0 else 0)
        #  return gd(self.n, self.batch_size)

        #  length = 0
        #  batches = [[
        #      len(s) if isinstance(s, Sampler) else 1
        #      for s in self.data[i:i + self.batch_size]
        #  ] for i in range(0, self.n - 1, self.batch_size)]
        #  return sum([min(b) for b in batches])
