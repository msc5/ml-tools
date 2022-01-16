import torch

from torchvision.transforms import ToTensor

from collections import defaultdict
from pathlib import Path
from json import load
from PIL import Image

from .tree import Tree
from .samplers import Sampler, BatchSampler


class Dataset:

    def __init__(
            self,
            directory: str,
    ):
        gpu_test = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if gpu_test else 'cpu')
        self.tree = self.build_from_dir(directory)
        self.path = Path(directory)

        # Default Settings
        self.transform = ToTensor()

        #  try:
        #      config = load(open('config.json'))['datasets'][self.name]
        #      self.structure = ['Root', *config['structure']]
        #      self.class_level = config['class_level'] + 1
        #  except:
        #      raise

    def build_from_dir(self, directory):
        """ Builds a Tree from a starting directory """
        def traverse(parent, depth):
            size = 1
            N_children = N_files = 0
            for path in parent.path.iterdir():
                if path.is_dir():
                    child = parent.add_child(path)
                    s, c, f = traverse(child, depth + 1)
                    size = s if s > size else size
                    N_children += c
                    N_files += f
                    self.levels['nodes'][depth + 1].append(child)
                else:
                    parent.add_file(path)
                    N_files += 1
                    self.levels['files'][depth + 1].append(path)

            # Set Default Samplers
            if parent.files:
                parent.sampler = Sampler(parent.files, {})
            else:
                samplers = [iter(c) for c in parent.get_children()]
                parent.sampler = Sampler(samplers, {})

            # Set Tree Info
            parent.info['size'] = size
            parent.info['depth'] = depth
            parent.info['N_children'] = N_children
            parent.info['N_files'] = N_files
            return size + 1, N_children + 1, N_files

        self.levels = {
            'nodes': defaultdict(list),
            'files': defaultdict(list)
        }
        path = Path(directory)
        root = Tree(path.name, path)
        traverse(root, 0)
        self.size = root.info['size'] + 1
        self.info = root.info
        return root

    def split(self, path):
        return self.tree.get(path)

    def collate(self, tensors):
        return torch.cat([x.unsqueeze(0) for x in tensors])

    def load(self, path):
        return self.transform(Image.open(path))
