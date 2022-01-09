import time

from collections import OrderedDict


class Timer:

    def __init__(self):
        self.times = OrderedDict()

    def __str__(self):
        msg = []
        for name, time in self.times.items():
            msg.append(f'{name:>30} : {time:4.5f} seconds')
        return '\n'.join(msg)

    def time(self, callback, label):
        start = time.perf_counter()
        val = callback()
        stop = time.perf_counter()
        self.times[label] = stop - start
        return val
