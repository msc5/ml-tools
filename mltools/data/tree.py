import os

from pathlib import Path
from collections.abc import Callable
from collections import defaultdict
from typing import Type


class Tree:

    def __init__(
        self,
        key: str,
        path: Path = None
    ):
        self.key = key
        self.path = path
        self.files = []
        self.children = {}
        self.info = defaultdict(int)

    def __iter__(self):
        """
        Returns an iterator over all children
        """
        return iter(self.sampler)

    def __len__(self):
        return len(self.sampler)

    def __lt__(self, other):
        return len(self) < len(other)

    def get_children(self):
        return list(self.children.values())

    def get(self, path):
        keys = path.split(os.sep)
        curr = self
        while keys:
            key = keys.pop(0)
            curr = curr.children[key]
        assert curr.key == key
        return curr

    def add_child(self, path):
        child = Tree(path.name, path)
        if child.key in self.children:
            raise KeyError
        self.info['n_children'] += 1
        self.children[child.key] = child
        return child

    def add_file(self, path):
        self.info['n_files'] += 1
        self.files.append(Path(path))

    def bfs(
        self,
        callback: Callable[['Tree', any], None] = None,
        params: any = None,
    ):
        """ Performs breadth-first search starting from this node """
        queue = [self]
        while queue:
            curr = queue.pop(0)
            if callback is not None:
                callback(curr, params)
            children = curr.children.items()
            for key, child in children:
                queue.append(child)

    def dfs(
        self,
        callback: Callable[['Tree', any], None] = None,
        params: any = None,
        order: 'pre' or 'post' = 'post',
    ):
        """ Performs depth-first search starting from this node """
        children = self.children.items()
        if callback is not None and order == 'pre':
            callback(self, params)
        for key, child in children:
            child.dfs(callback, None if not params else params.copy(), order)
        if callback is not None and order == 'post':
            callback(self, params)
