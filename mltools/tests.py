
from .data.tree import Tree
from .data.samplers import Sampler, BatchSampler
from .data.dataset import Dataset
from .data.loaders import FewShotLoader
from .util import Timer
from rich import print
from rich.progress import track

if __name__ == '__main__':

    #  def s(s, n): return Sampler([s + ' -> ' + str(i) for i in range(n)])

    #  def b(s, n, m): return BatchSampler(
    #      ['b' + s + ' -> ' + str(i) for i in range(n)],
    #      {'batch_size': m, 'batch_mode': 'wrap'})

    #  bs = BatchSampler(
    #      [b(str(i), 10, 3) for i in range(4)],
    #      {'batch_size': 3, 'batch_mode': 'wrap'}
    #  )
    #  for i, x in enumerate(bs):
    #      print(i, x)

    #  timer = Timer()

    #  def iterate(iterable, show=False):
    #      print('Iterating: ', iterable)
    #      print('__len__(): ', len(iterable))
    #      for i, x in enumerate(track(iterable)):
    #          print(x)
    #          pass
    #      #  for i, x in enumerate(iterable):
    #      #      if show:
    #      #          if i % 2000 == 0:
    #      #              print(f'{i:<4}', x)
    #      print('Number of Iterations: ', i + 1)

    #  path = 'datasets/miniimagenet'
    path = 'datasets/dummy'
    #  path = 'datasets/omniglot'

    dataset = Dataset(path)
    train = dataset.split('train')
    test = dataset.split('test')
    sampler = FewShotLoader(dataset, {'k': 5, 'n': 1, 'batch_size': 1})

    for i, x in enumerate(sampler):
        #  print(i, x)
        pass

    #  print(next(sampler))
    #  print(next(sampler))
    #  print(next(sampler))

    #  iterate(sampler)

    #  print(timer)
