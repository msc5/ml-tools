import os
import collections

from pathlib import Path
from collections.abc import Callable
from typing import Type
from rich.tree import Tree as rTree
from rich.console import Console

from .samplers import Sampler, BatchSampler


class TreeNode:

    def __init__(
        self,
        key: str,
        path: Path = None
    ):
        self.key = key
        self.path = path
        self.files = []
        self.children = {}
        self.info = collections.defaultdict(int)

    def __iter__(self):
        """
        Returns an iterator over all children
        """
        return iter(self.sampler)

    def __str__(self):
        """
        Returns a string representation of the TreeNode in DFS order
        """
        def callback(node, params):
            if 'tree' not in params:
                params['root'] = params['tree'] = rTree('[cyan]' + node.key)
            else:
                params['tree'] = params['tree'].add(node.key)
            if node.files:
                for f in node.files:
                    params['tree'].add(str(f.name))
        p = {}
        self.dfs(callback, p, order='pre')
        c = Console()
        with c.capture() as capture:
            c.print(p['root'])
        return capture.get()

    def __len__(self):
        return len(self.sampler)

    def __lt__(self, other):
        return len(self) < len(other)

    def get_children(self):
        return list(self.children.values())

    def add_child(self, path):
        child = TreeNode(path.name, path)
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
        callback: Callable[['TreeNode', any], None] = None,
        params: any = None,
    ):
        """
        Performs breadth-first search starting from this node
        """
        queue = [self]
        while queue:
            curr = queue.pop(0)
            if callback is not None:
                callback(curr, params)
            children = curr.children.items()
            for key, child in sorted(children):
                queue.append(child)

    def dfs(
        self,
        callback: Callable[['TreeNode', any], None] = None,
        params: any = None,
        order: 'pre' or 'post' = 'post',
    ):
        """
        Performs depth-first search starting from this node
        """
        children = self.children.items()
        if callback is not None and order == 'pre':
            callback(self, params)
        for key, child in sorted(children):
            child.dfs(callback, None if not params else params.copy(), order)
        if callback is not None and order == 'post':
            callback(self, params)


class Tree:

    def __init__(self, directory):
        self.path = Path(directory)
        self.root = TreeNode(self.path.name, self.path)
        self.build_from_dir()

    def __iter__(self):
        yield from self.root

    def __len__(self):
        return len(self.root)

    def get(self, path):
        keys = path.split(os.sep)
        curr = self.root
        while keys:
            curr = curr.children[keys.pop(0)]
        return curr

    def put_samplers(self, samplers):
        def put_sampler(node, params):
            depth = node.info['depth']
            if depth in samplers:
                sampler, callback, params = samplers[depth]
            else:
                sampler = Sampler
                def callback(x): return x
                params = {}
            if node.files:
                data = node.files
            else:
                data = list(node.children.values())
            node.sampler = sampler(data, params)
            node.callback = callback
            node.info['n_iter'] = len(node.sampler)
        self.dfs(put_sampler, mode='postorder')
        return self

    def bfs(
        self,
        callback: Callable[['TreeNode', any], None] = None,
        params: any = None,
    ):
        self.root.bfs(callback, params)

    def dfs(
        self,
        callback: Callable[['TreeNode', any], None] = None,
        params: any = None,
        order: 'pre' or 'in' or 'post' = 'post',
    ):
        self.root.dfs(callback, params)

    def build_from_dir(self):
        """
        Builds a Tree from a starting directory
        """

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
                    self.levels['nodes'][depth + 1] += 1
                else:
                    parent.add_file(path)
                    N_files += 1
                    self.levels['files'][depth + 1] += 1

            # Set Default Samplers
            if parent.files:
                parent.sampler = Sampler(parent.files, {})
            else:
                samplers = [iter(c) for c in parent.get_children()]
                parent.sampler = Sampler(samplers, {})

            # Set TreeNode Info
            parent.info['size'] = size
            parent.info['depth'] = depth
            parent.info['N_children'] = N_children
            parent.info['N_files'] = N_files
            return size + 1, N_children + 1, N_files

        self.levels = {
            'nodes': collections.defaultdict(int),
            'files': collections.defaultdict(int)
        }
        traverse(self.root, 0)
        self.size = self.root.info['size'] + 1
        self.levels['nodes'][0] = 1
        self.levels = {
            k: [v[d] for d in range(self.size)]
            for k, v in self.levels.items()
        }
