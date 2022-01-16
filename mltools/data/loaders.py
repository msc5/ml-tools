from .dataset import Dataset
from .samplers import Sampler, BatchSampler


class FewShotLoader:

    def __init__(self, dataset: Dataset, params: object = None):
        self.dataset = dataset
        self.params = params
        class_level = dataset.info['size'] - 1
        self.classes = dataset.levels['nodes'][class_level]
        k = params.get('k',  1)
        n = params.get('n',  1)
        m = params.get('m',  1)
        self.subsamplers = [
            BatchSampler(
                node.files, {'batch_size': n + m, 'batch_mode': 'random'}
            ) for node in self.classes
        ]
        self.sampler = BatchSampler(
            self.subsamplers, {'batch_size': k, 'batch_mode': 'random'})

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)
