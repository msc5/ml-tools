import random
import pprint
import math


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
    batch_size = params.get('batch_size', 1)
    it_data = iter(data)
    iters = [next(it_data) for _ in range(batch_size)]
    while True:
        vals = []
        for i, it in enumerate(iters):
            try:
                vals.append(next(it))
            except StopIteration:
                try:
                    iters[i] = next(it_data)
                except StopIteration:
                    return
                vals.append(next(iters[i]))
        yield vals


def RandomBatchSampler(data, params):
    yield from BatchSampler(
        RandomSampler(data, params),
        params
    )


if __name__ == '__main__':

    pp = pprint.PrettyPrinter(width=200)

    m = (5, 20)
    data = [
        [
            f'{j - m[0] + 1:3.0f}_{i:<3.0f}' for i in range(j)
        ] for j in range(*m)
    ]
    pp.pprint([[a for a in b] for b in data])
    sampler = ParallelSampler(iter(iter(d)
                              for d in data), {'batch_size': m[0]})

    for i, x in enumerate(sampler):
        print(f'{i:3}', x)
    n_data = (m[1] - m[0]) * (m[0] + math.ceil(m[0] / 2))
    n_seen = (i + 1) * m[0]
    n_norm = m[0]**2
    print('# Data: ', n_data)
    print('# Seen: ', n_seen)
    print('# Norm: ', n_norm)
    print('Permuted: ', n_seen / n_data, '%')
