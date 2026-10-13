"""The factor expression language: a whitelist over Python syntax.

Example: rank(ts_mean(volume, 5) / ts_mean(volume, 60))

Expressions are parsed with Python's `ast` module and checked node by node. Only listed
operators, fields and numeric constants are allowed; window lengths must be literal
positive integers. Because no operator can shift data backwards in time (there is no
lead or negative delay), a valid expression cannot read the future.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from timefence.core.errors import ExpressionError
from timefence.factors import ops

FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "returns")
MAX_WINDOW = 250
MAX_DEPTH = 7
MAX_NODES = 40
MAX_LOOKBACK = 300


@dataclass(frozen=True)
class OpSpec:
    func: Callable
    n_series: int  # how many expression arguments
    windowed: bool  # whether a trailing integer window follows
    doc: str
    cross_sectional: bool = False  # compares stocks with each other on the same day


OPS: dict[str, OpSpec] = {
    "delay": OpSpec(ops.delay, 1, True, "value d days ago"),
    "delta": OpSpec(ops.delta, 1, True, "x - delay(x, d)"),
    "pct_change": OpSpec(ops.pct_change, 1, True, "x / delay(x, d) - 1"),
    "ts_mean": OpSpec(ops.ts_mean, 1, True, "mean over the last d days"),
    "ts_sum": OpSpec(ops.ts_sum, 1, True, "sum over the last d days"),
    "ts_std": OpSpec(ops.ts_std, 1, True, "standard deviation over the last d days"),
    "ts_max": OpSpec(ops.ts_max, 1, True, "max over the last d days"),
    "ts_min": OpSpec(ops.ts_min, 1, True, "min over the last d days"),
    "ts_rank": OpSpec(ops.ts_rank, 1, True, "percentile of today within the last d days"),
    "ts_zscore": OpSpec(ops.ts_zscore, 1, True, "(x - ts_mean) / ts_std over d days"),
    "decay_linear": OpSpec(
        ops.decay_linear, 1, True, "linearly weighted mean, recent days weigh more"
    ),
    "ts_corr": OpSpec(ops.ts_corr, 2, True, "rolling correlation of x and y over d days"),
    "ts_cov": OpSpec(ops.ts_cov, 2, True, "rolling covariance of x and y over d days"),
    "rank": OpSpec(ops.rank, 1, False, "percentile rank across universe stocks on each day", True),
    "zscore": OpSpec(ops.zscore, 1, False, "z-score across universe stocks on each day", True),
    "abs": OpSpec(ops.abs_, 1, False, "absolute value"),
    "sign": OpSpec(ops.sign, 1, False, "sign: -1, 0 or 1"),
    "log": OpSpec(ops.log, 1, False, "signed log: sign(x) * log(1 + |x|)"),
    "sqrt": OpSpec(ops.sqrt, 1, False, "signed square root"),
    "max": OpSpec(ops.max_, 2, False, "element-wise max"),
    "min": OpSpec(ops.min_, 2, False, "element-wise min"),
}

BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


@dataclass(frozen=True)
class Parsed:
    """A validated expression."""

    canonical: str  # normalised text, used as the factor's identity
    tree: ast.Expression
    lookback: int  # trading days of history needed before the first valid value
    fields: frozenset[str]
    n_nodes: int
    depth: int


def describe_language() -> str:
    """Operator reference given to the research agent."""
    lines = [
        f"Fields: {', '.join(FIELDS)} (prices are dividend-adjusted; returns = daily close-to-close)."
    ]
    lines.append("Arithmetic: + - * / and numeric constants.")
    for name, spec in OPS.items():
        args = ", ".join(["x", "y"][: spec.n_series] + (["d"] if spec.windowed else []))
        lines.append(f"{name}({args}): {spec.doc}")
    lines.append(f"Windows d are literal integers from 1 to {MAX_WINDOW}.")
    lines.append(f"At most {MAX_NODES} nodes and depth {MAX_DEPTH}.")
    return "\n".join(lines)


def parse(text: str) -> Parsed:
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("expression is empty")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error: {exc.msg}") from exc

    checker = _Checker()
    lookback, depth, _ = checker.visit(tree.body)
    if not checker.fields:
        raise ExpressionError("expression must use at least one data field")
    if checker.n_nodes > MAX_NODES:
        raise ExpressionError(f"expression has {checker.n_nodes} nodes, limit is {MAX_NODES}")
    if depth > MAX_DEPTH:
        raise ExpressionError(f"expression depth is {depth}, limit is {MAX_DEPTH}")
    if lookback > MAX_LOOKBACK:
        raise ExpressionError(f"needs {lookback} days of history, limit is {MAX_LOOKBACK}")
    return Parsed(
        canonical=ast.unparse(tree.body),
        tree=tree,
        lookback=lookback,
        fields=frozenset(checker.fields),
        n_nodes=checker.n_nodes,
        depth=depth,
    )


class _Checker:
    """Walks the tree, rejects anything not whitelisted, and tracks history and depth."""

    def __init__(self) -> None:
        self.fields: set[str] = set()
        self.n_nodes = 0

    def visit(self, node: ast.AST) -> tuple[int, int, bool]:
        """Returns (lookback days, depth, whether a data field is used) of the sub-expression."""
        self.n_nodes += 1
        if isinstance(node, ast.Name):
            if node.id not in FIELDS:
                raise ExpressionError(f"unknown field '{node.id}'; fields are {', '.join(FIELDS)}")
            self.fields.add(node.id)
            return (1 if node.id == "returns" else 0), 1, True
        if isinstance(node, ast.Constant):
            value = node.value
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ExpressionError(f"constant {value!r} is not a finite number")
            return 0, 1, False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            lookback, depth, has_field = self.visit(node.operand)
            return lookback, depth + 1, has_field
        if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            right = node.right
            if (
                isinstance(node.op, ast.Div)
                and isinstance(right, ast.Constant)
                and right.value == 0
            ):
                raise ExpressionError("division by zero")
            lb1, d1, f1 = self.visit(node.left)
            lb2, d2, f2 = self.visit(node.right)
            return max(lb1, lb2), max(d1, d2) + 1, f1 or f2
        if isinstance(node, ast.Call):
            return self._visit_call(node)
        raise ExpressionError(f"'{ast.unparse(node)}' is not allowed")

    def _visit_call(self, node: ast.Call) -> tuple[int, int, bool]:
        if not isinstance(node.func, ast.Name) or node.func.id not in OPS:
            name = ast.unparse(node.func)
            raise ExpressionError(f"unknown operator '{name}'; see the operator list")
        name = node.func.id
        if node.keywords:
            raise ExpressionError(f"{name}: keyword arguments are not allowed")
        spec = OPS[name]
        expected = spec.n_series + (1 if spec.windowed else 0)
        if len(node.args) != expected:
            raise ExpressionError(f"{name} takes {expected} arguments, got {len(node.args)}")

        window = 0
        if spec.windowed:
            arg = node.args[-1]
            if not (isinstance(arg, ast.Constant) and type(arg.value) is int):
                raise ExpressionError(f"{name}: window must be a literal integer")
            window = arg.value
            if not 1 <= window <= MAX_WINDOW:
                raise ExpressionError(f"{name}: window {window} outside 1..{MAX_WINDOW}")
            self.n_nodes += 1

        lookback, depth, uses = 0, 0, []
        for arg in node.args[: spec.n_series]:
            lb, d, has_field = self.visit(arg)
            lookback, depth = max(lookback, lb), max(depth, d)
            uses.append(has_field)
        # max/min may compare a series with a constant; every other operator needs series.
        ok = any(uses) if name in ("max", "min") else all(uses)
        if not ok:
            raise ExpressionError(f"{name}: arguments must be built from data fields")
        return lookback + window, depth + 1, True


def evaluate(
    parsed: Parsed,
    field: Callable[[str], pd.DataFrame],
    universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute a parsed expression. `field(name)` returns the panel for a data field.

    Cross-sectional operators (rank, zscore) compare only stocks in `universe` on each day.
    That keeps them point in time: which other stocks happen to be in the panel (for
    example, ones that only become liquid years later) cannot change a value.
    """
    result = _eval(parsed.tree.body, field, universe)
    if not isinstance(result, pd.DataFrame):
        raise ExpressionError("expression evaluates to a constant, not a factor")
    return ops._clean(result.astype(float))


def _eval(node: ast.AST, field: Callable[[str], pd.DataFrame], universe: pd.DataFrame | None):
    if isinstance(node, ast.Name):
        return field(node.id)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, field, universe)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left, field, universe), _eval(node.right, field, universe)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if (
            not isinstance(left, pd.DataFrame)
            and not isinstance(right, pd.DataFrame)
            and right == 0
        ):
            raise ExpressionError("division by a zero constant")
        quotient = left / right
        return ops._clean(quotient) if isinstance(quotient, pd.DataFrame) else quotient
    if isinstance(node, ast.Call):
        spec = OPS[node.func.id]
        series = [_eval(a, field, universe) for a in node.args[: spec.n_series]]
        if spec.cross_sectional and universe is not None:
            series = [s.where(universe) for s in series]
        if spec.windowed:
            return spec.func(*series, node.args[-1].value)
        return spec.func(*series)
    raise ExpressionError(f"cannot evaluate '{ast.unparse(node)}'")
