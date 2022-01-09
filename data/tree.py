import os
import pprint
import collections

from pathlib import Path


class color:
    PURPLE = '\033[1;35;48m'
    CYAN = '\033[1;36;48m'
    BOLD = '\033[1;37;48m'
    BLUE = '\033[1;34;48m'
    GREEN = '\033[1;32;48m'
    YELLOW = '\033[1;33;48m'
    RED = '\033[1;31;48m'
    BLACK = '\033[1;30;48m'
    UNDERLINE = '\033[4;37;48m'
    END = '\033[1;37;0m'


class TreeNode:

    def __init__(self, key, path=None):
        self.key = key
        self.path = path
        self.files = []
        self.children = collections.OrderedDict()

    def add_child(self, child):
        self.children[child.key] = child

    def __iter__(self):
        pass

    def __next__(self):
        pass


class Tree:

    def __init__(self, directory):
        self.path = Path(directory)
        self.root = TreeNode(self.path.name, self.path)
        self.build_from_dir()

    def __str__(self):
        """
        Returns a string representation of the Tree in DFS order
        """

        def space(v, n): return f'{"":>{n * 5}}{v}'

        def callback(node, params):
            msg.append(space(node.key, params['depth']))
            params['depth'] += 1

        msg = []
        self.dfs(callback, {'depth': 0})
        return '\n'.join(msg)

    def bfs(self, callback=None, params={}):
        queue = [self.root]
        while queue:
            curr = queue.pop(0)
            if callback:
                callback(curr, params)
            for key, child in curr.children.items():
                queue.append(child)

    def dfs(self, callback=None, params={}):
        def traverse(node, params):
            if callback:
                callback(node, params)
            for key, child in node.children.items():
                traverse(child, params.copy())
        traverse(self.root, params)

    def build_from_dir(self):
        def traverse(parent: TreeNode):
            for path in parent.path.iterdir():
                if path.is_dir():
                    child_node = TreeNode(path.name, path)
                    parent.add_child(child_node)
                    traverse(child_node)
                else:
                    parent.files.append(Path(path))
        traverse(self.root)


if __name__ == '__main__':

    #  directory = 'datasets/miniimagenet'
    #  directory = 'datasets/omniglot'
    directory = 'datasets/dummy'

    tree = Tree(directory)
    #  print(tree)
