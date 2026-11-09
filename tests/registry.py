import itertools
from collections.abc import Callable
from dataclasses import dataclass

CASES = []
BENCHES = []


class Skip(Exception):
    pass


@dataclass
class Entry:
    id: str
    name: str
    fn: Callable
    params: dict


def _label(value):
    if isinstance(value, dict):
        return ",".join(f"{k}={v}" for k, v in value.items()) or "default"
    if isinstance(value, tuple):
        return "x".join(map(str, value))
    return str(value)


def _register(table, name, where, grid):
    def decorate(fn):
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, values))
            if where is not None and not where(**params):
                continue
            tag = "-".join(f"{k}={_label(v)}" for k, v in params.items())
            table.append(Entry(f"{name}[{tag}]" if tag else name, name, fn, params))
        return fn

    return decorate


def case(name, where=None, **grid):
    """Register a test over the cartesian product of the keyword lists in ``grid``.

    The decorated function is called as ``fn(ctx, **params)`` once per combination.

    Args:
        name: Test name; each combination is listed as ``name[key=value-...]``.
        where: Optional predicate over the parameters; combinations it rejects are dropped.
        **grid: Lists of values per parameter.

    Returns:
        The decorator.
    """
    return _register(CASES, name, where, grid)


def bench(name, where=None, **grid):
    """Register a benchmark like ``case``; it only runs with ``--bench``.

    Args:
        name: Benchmark name.
        where: Optional predicate over the parameters.
        **grid: Lists of values per parameter.

    Returns:
        The decorator.
    """
    return _register(BENCHES, name, where, grid)
