import os
import collections

from pathlib import Path

from .samplers import Sampler


class TreeNode:

    def __init__(self, key, path=None):
        self.key = key
        self.path = path
        self.files = []
        self.children = {}
        self.info = collections.defaultdict(int)

    def __len__(self):
        if self.sampler:
            return len(self.sampler)
        else:
            return len(self.children)

    def __iter__(self):
        if not hasattr(self, 'sampler'):
            self.sampler = Sampler(self.children.values(), {})
        for sample in self.sampler:
            if self.files:
                yield self.callback(sample)
            else:
                if isinstance(sample, TreeNode):
                    yield from sample
                elif isinstance(sample, list):
                    yield self.callback(sample)

    def __lt__(self, other):
        return len(other) < len(self)

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

    def bfs(self, callback=None, params={}):
        queue = [self]
        while queue:
            curr = queue.pop(0)
            if callback:
                callback(curr, params)
            children = curr.children.items()
            for key, child in sorted(children):
                queue.append(child)

    def dfs(self, callback=None, params={}, mode='preorder'):
        def traverse(node, params):
            if callback and mode == 'preorder':
                callback(node, params)
            children = node.children.items()
            for key, child in sorted(children):
                traverse(child, params.copy())
            if callback and mode == 'postorder':
                callback(node, params)
        traverse(self, params)


class Tree:

    def __init__(self, directory):
        self.path = Path(directory)
        self.root = TreeNode(self.path.name, self.path)
        self.build_from_dir()

    #  def __str__(self):
    #      """
    #      Returns a string representation of the Tree in DFS order
    #      """

    #      def callback(node, params):
    #          params['tree'] = params['tree'].add(node.key)
    #          if node.files:
    #              for f in node.files:
    #                  params['tree'].add(str(f))

    #      tree = rTree('Dataset')
    #      self.dfs(callback, {'tree': tree})
    #      return tree

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
            #  node.callback = callback
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

    #  def bfs(self, callback=None, params={}):
    #      queue = [self.root]
    #      while queue:
    #          curr = queue.pop(0)
    #          if callback:
    #              callback(curr, params)
    #          children = curr.children.items()
    #          for key, child in sorted(children):
    #              queue.append(child)

    #  def dfs(self, callback=None, params={}, mode='preorder'):
    #      def traverse(node, params):
    #          if callback and mode == 'preorder':
    #              callback(node, params)
    #          children = node.children.items()
    #          for key, child in sorted(children):
    #              traverse(child, params.copy())
    #          if callback and mode == 'postorder':
    #              callback(node, params)
    #      traverse(self.root, params)

    def bfs(self, callback=None, params={}):
        self.root.bfs(callback, params)

    def dfs(self, callback=None, params={}, mode='preorder'):
        self.root.dfs(callback, params, mode)

    def build_from_dir(self):
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
            k: [
                v[d] for d in range(self.size)
            ] for k, v in self.levels.items()
        }
