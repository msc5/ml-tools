import torch

from rich.progress import track
from rich import print

from .dataset import Dataset
from .samplers import BatchSampler


class FewShotDataset (Dataset):

    def __init__(
            self,
            directory: str,
            params: object,
    ):
        super().__init__(directory)
        assert 'k' in params
        assert 'n' in params
        assert 'm' in params
        k, n, m = params.get('k', 1), params.get('n', 1), params.get('m', 1)
        self.put_samplers({
            self.class_level - 1: {
                'sampler': BatchSampler,
                'batch_size': k,
                'batch_mode': 'random',
                'callback': self.collate
            },
            self.class_level: {
                'sampler': BatchSampler,
                'batch_size': n + m,
                'batch_mode': 'random',
                'callback': self.sample
            }
        })

    def sample(self, images):
        return self.collate([self.load(i) for i in images])
