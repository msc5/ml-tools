from torch.utils.data import RandomSampler, BatchSampler

# from .tree import Tree


class NestedIterator:

    def __init__(self, *generators):
        assert generators is not None
        self.generators = generators
        self.size = len(generators)
        self.len = sum([1 for _ in iter(self)])

    def __iter__(self):
        next_gens = iter(self.generators)
        last_gens = iter(self.generators)
        next_gen = next(next_gens)
        next_node = NestedIteratorNode(next_gen)
        while True:
            last_node = next_node
            next_gen = next(next_gens, None)
            if not next_gen:
                break
            next_node = NestedIteratorNode(next_gen)
            last_node.add(next_node)
            last_node.step()
        self.bottom = next_node
        self.top = next_node.top()
        return self

    def __next__(self):
        return next(self.bottom)

    def __str__(self):
        msg = []

        def cb(node):
            msg.append('\n'.join([
                (f'{"Node":15}' + str(node)),
                (f'{"Value":15}' + str(node.val)),
                (f'{"Iterator":15}' + str(node.it)),
                (f'{"Generator":15}' + str(node.gen))
            ]))
        self.top.bottom(cb)
        return '\n'.join(msg)

    def __len__(self):
        return self.len


class NestedIteratorNode:

    def __init__(self, generator, up=None, down=None):
        assert generator
        self.val = None
        self.gen = generator
        self.it = iter(generator)
        self.up = up
        self.down = down

    def __iter__(self):
        return self

    def __next__(self):
        self.step()
        top = self.top()
        vals = []
        def cb(node): vals.append(node.val)
        top.bottom(cb)
        return vals

    def top(self, cb=None):
        curr = self
        while curr.up:
            if cb:
                cb(curr)
            curr = curr.up
        if cb:
            cb(curr)
        return curr

    def bottom(self, cb=None):
        curr = self
        while curr.down:
            if cb:
                cb(curr)
            curr = curr.down
        if cb:
            cb(curr)
        return curr

    def add(self, node):
        bottom = self.bottom()
        bottom.down = node
        node.up = bottom
        return node

    def step(self):
        val = next(self.it, None)
        if val is None:
            if self.up:
                self.up.step()
                self.it = iter(self.gen)
                self.val = next(self.it)
            else:
                raise StopIteration
        else:
            self.val = val


class ParallelIterator:

    def __init__(self, generator, n):
        self.gen = generator
        self.len = sum([1 for _ in iter(self.gen)])
        self.n = n

    def __iter__(self):
        self.iters = [iter(self.gen) for _ in range(self.n)]
        return self

    def __next__(self):
        vals = [next(it) for it in self.iters]
        return vals

    def __len__(self):
        return self.len


if __name__ == '__main__':

    data = [str(i) for i in range(6)]

    test_tree = Tree(directory='test')

    for i, t in test_tree:
        print(f'{i:<4}', t)
    print('Length: ', i + 1)
