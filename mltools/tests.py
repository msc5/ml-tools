
from .data.tree import Tree
from .data.samplers import Sampler, BatchSampler
from .data.dataset import Dataset
from .data.loaders import FewShotLoader
from .util import Timer
from rich import print
from rich.progress import track

if __name__ == '__main__':

    def s(n, s): return Sampler([s + ' -> ' + str(i) for i in range(n)])
    bs = BatchSampler(
        [s(10, str(i)) for i in range(6)],
        {'batch_size': 3, 'batch_mode': 'wrap'}
    )
    for x in bs:
        print(x)

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

    #  #  path = 'datasets/miniimagenet'
    #  #  path = 'datasets/dummy'
    #  path = 'datasets/omniglot'

    #  dataset = Dataset(path)
    #  train = dataset.split('train')
    #  test = dataset.split('test')
    #  sampler = FewShotLoader(dataset, {'k': 5, 'n': 1, 'm': 1})
    #  iterate(sampler)

    #  print(timer)
