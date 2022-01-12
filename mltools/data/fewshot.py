import torch

from rich.progress import track
from rich import print

from .dataset import Dataset
from .samplers import BatchSampler, ParallelSampler


class FewShotDataset (Dataset):

    def __init__(
            self,
            directory: str,
            device: torch.device
    ):
        super().__init__(directory, device)

    def sample(self, images):
        return self.collate([self.load(i) for i in images])

    def get_iter(self, params):
        """
        Returns an iterable over images in the dataset with the
        specified parameters
        """
        split = params.get('split', None)
        k, n, m = params.get('k', 1), params.get('n', 1), params.get('m', 1)
        self.tree.put_samplers({
            self.class_level - 1: (
                ParallelSampler,
                self.collate,
                {'batch_size': k}
            ),
            self.class_level: (
                BatchSampler,
                self.sample,
                {'batch_size': n + m}
            )
        })
        if split:
            self.split = self.tree.get(split)
        return self
