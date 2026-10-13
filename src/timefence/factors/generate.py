"""Random factor expressions: the baseline the agent has to beat, and inputs for tests."""

from __future__ import annotations

import random

from timefence.core.errors import ExpressionError
from timefence.factors import dsl

WINDOWS = (2, 3, 5, 10, 20, 40, 60, 120)
TS_UNARY = tuple(n for n, s in dsl.OPS.items() if s.windowed and s.n_series == 1)
TS_BINARY = tuple(n for n, s in dsl.OPS.items() if s.windowed and s.n_series == 2)
PLAIN_UNARY = tuple(n for n, s in dsl.OPS.items() if not s.windowed and s.n_series == 1)


def random_expression(rng: random.Random, max_depth: int = 4) -> str:
    """A random expression the DSL accepts, in canonical form."""
    for _ in range(200):
        try:
            return dsl.parse(_grow(rng, 0, max_depth)).canonical
        except ExpressionError:
            continue
    raise RuntimeError("could not generate a valid expression")


def _grow(rng: random.Random, depth: int, max_depth: int) -> str:
    if depth >= max_depth or (depth > 0 and rng.random() < 0.3):
        return rng.choice(dsl.FIELDS)
    nxt = depth + 1
    roll = rng.random()
    if roll < 0.45:
        return f"{rng.choice(TS_UNARY)}({_grow(rng, nxt, max_depth)}, {rng.choice(WINDOWS)})"
    if roll < 0.55:
        window = rng.choice(WINDOWS[2:])
        x, y = _grow(rng, nxt, max_depth), _grow(rng, nxt, max_depth)
        return f"{rng.choice(TS_BINARY)}({x}, {y}, {window})"
    if roll < 0.70:
        return f"{rng.choice(PLAIN_UNARY)}({_grow(rng, nxt, max_depth)})"
    if roll < 0.95:
        op = rng.choice("+-*/")
        return f"({_grow(rng, nxt, max_depth)} {op} {_grow(rng, nxt, max_depth)})"
    return f"-{_grow(rng, nxt, max_depth)}"
