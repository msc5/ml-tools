import random


def SimpleSampler(data, params):
    yield from data


def RandomSampler(data, params):
    s = data.copy()
    random.shuffle(s)
    yield from s


def BatchSampler(data, params):
    batch_size = params.get('batch_size', 1)
    keep_last = params.get('keep_last', False)
    try:
        while True:
            val = []
            for i in range(batch_size):
                val.append(next(data))
            yield val
    except StopIteration:
        if keep_last:
            yield val


def RandomBatchSampler(data, params):
    yield from BatchSampler(
        RandomSampler(data, params),
        params
    )
