import os
import itertools


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


def section(msg):
    n, _ = os.get_terminal_size()
    l = len(msg)
    print('')
    print(msg + ' ' + '-'*(n - 1 - l))
    print('')


def tabulate(*entries):

    def format(val, width):
        space = ' ' * width
        msg = ''
        if val is None:
            msg = space
        if isinstance(val, str) or type(val) == int:
            msg = (
                f'{val:<{width}}'
                if val is not None else space
            )
        if type(val) == float:
            msg = (
                f'{val:<{width}.5f}'
                if val is not None else space
            )
        else:
            msg = (
                f'{str(val):<{width}}'
                if str(val) is not None else space
            )
        return msg

    widths = [max(
        len(t),
        max([len(str(val)) for val in v])
    ) + 5 for t, v in entries]
    titles, data = zip(*entries)
    table = itertools.zip_longest(*data)

    msg = []
    msg.append([format(t, w) for t, w in zip(titles, widths)])
    for value in table:
        msg.append([
            format(v, w) for v, w in zip(value, widths)
        ])

    return '\n'.join([''.join(v) for v in msg])
