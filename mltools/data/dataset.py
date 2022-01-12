import torch

from torchvision.transforms import ToTensor

from pathlib import Path
from json import load
from PIL import Image

from ..util import tabulate
from .tree import Tree


class Dataset:

    def __init__(
            self,
            directory: str,
            device: torch.device
    ):
        self.device = device
        self.tree = Tree(directory)
        self.split = self.tree.root
        self.transform = ToTensor()
        self.path = Path(directory)
        self.name = self.path.name
        try:
            config = load(open('config.json'))['datasets'][self.name]
            self.structure = ['Root', *config['structure']]
            self.class_level = config['class_level'] + 1
        except:
            print('Error Opening datasets.json')
            raise

    def __str__(self):
        """
        Returns a string representation of the dataset
        """
        size = self.tree.root.info['size']
        return self.name + '\n' + tabulate(
            ('Depth', range(size + 1)),
            ('Name', self.structure),
            ('Folder Count', self.tree.levels['nodes']),
            ('File Count', self.tree.levels['files']),
        )

    def __iter__(self):
        """
        Returns an iterable over the files in the dataset
        """
        return iter(self.split)

    def __len__(self):
        """
        Returns the length of the dataset's total iteration
        """
        return len(self.split)

    def info(self):
        return self.split.info

    def load_image(self, tree):
        """
        Takes a tree as input and returns the file associated with its path
        """
        image = Image.open(tree.val)
        return self.transform(image).to(device)

    def split(self, directory):
        assert directory is not None
        split = self.tree.get(directory)
        return split.generator(self.load_image)

    def collate(self, tensors):
        return torch.cat([x.unsqueeze(0) for x in tensors])

    def load(self, path):
        return self.transform(Image.open(path))
