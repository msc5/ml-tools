import random
import pprint


def SimpleSampler(data, params=None):
    yield from data


def RandomSampler(data, params=None):
    s = data.copy()
    random.shuffle(s)
    yield from s


def BatchSampler(data, params):
    batch_size = params.get('batch_size', 1)
    keep_last = params.get('keep_last', False)
    full_permute = params.get('full_permute', False)
    try:
        i = 0
        while True:
            val = []
            for _ in range(batch_size):
                val.append(next(data))
            if i == 0:
                firstval = val
                i += 1
            yield val
    except StopIteration:
        if keep_last:
            yield val
        elif full_permute:
            yield [*val, *firstval[len(val):]]

def ParallelSampler(data, params):
    pass


def RandomBatchSampler(data, params):
    yield from BatchSampler(
        RandomSampler(data, params),
        params
    )


if __name__ == '__main__':

    pp = pprint.PrettyPrinter()

    n = 30
    bs = 4
    data = [f'{i:3.0f}' for i in range(n)]
    sampler = RandomBatchSampler(data, {'batch_size': bs})

    print(data)
    for i, x in enumerate(sampler):
        pp.pprint(x)
    print('Permuted: ', 100 * ((i + 1) * bs) / n, '%')
