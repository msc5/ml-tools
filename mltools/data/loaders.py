import torch
import random
import itertools

from PIL import Image
from torchvision.transforms import ToTensor

from .dataset import Dataset
from .samplers import Sampler, BatchSampler


class FewShotLoader:

    def __init__(self, dataset: Dataset, params: object = None):

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.params = params
        self.transform = ToTensor()

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
        for c, e in itertools.product(self.c, self.e):
            if not self.visited[c][e]:
                self.visited[c][e] = True
                self.v += 1
        idx = [[self.examples[c][e] for e in self.e] for c in self.c]
        img = [[self.load(e) for e in c] for c in idx]
        s = self.collate([e[:self.n] for e in img])
        q = self.collate([e[self.n:] for e in img])
        label = torch.eye(self.k).repeat_interleave(
            self.m, dim=0).to(self.device)
        return s, q, label

    def load(self, img):
        return self.transform(Image.open(img).resize((28, 28)))

    def collate(self, img):
        def cat(l): return torch.cat([c.unsqueeze(0) for c in l])
        return cat([cat(i) for i in img]).to(self.device)