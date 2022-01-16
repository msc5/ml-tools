import random

from .dataset import Dataset
from .samplers import Sampler, BatchSampler


class FewShotLoader:

    def __init__(self, dataset: Dataset, params: object = None):
        self.dataset = dataset
        self.params = params

        class_level = dataset.info['size'] - 1
        self.k = params.get('k',  1)
        self.n = params.get('n',  1)
        self.m = params.get('batch_size', 1)
        self.b = self.n + self.m

        self.classes = dataset.levels['nodes'][class_level]
        self.examples = [n.files for n in self.classes]
        self.visited = [[False for e in c] for c in self.examples]
        self.v = 0
        self.C = len(self.examples)
        self.E = len(self.examples[0])

    def __iter__(self):
        self.v = 0
        self.visited = [[False for e in c] for c in self.examples]
        return self

    def __next__(self):
        self.c = random.sample(range(self.C), self.k)
        self.e = random.sample(range(self.E), self.b)
        for c in self.c:
            for e in self.e:
                if not self.visited[c][e]:
                    self.visited[c][e] = True
                    self.v += 1
        print(self.v / (self.C * self.E))
        return [[self.examples[c][e] for e in self.e] for c in self.c]
