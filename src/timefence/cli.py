"""Command line: `timefence data ...`, `timefence run ...`, `timefence compare ...`, `timefence audit ...`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from timefence.agents.llm import PROVIDERS
from timefence.data.store import MarketStore


def _fmt(value, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Copy KEY=VALUE lines from a local .env file into the environment.

    Variables that are already set win, so a key exported in the shell overrides the file.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _write_meta(root: Path, info: dict) -> None:
    info = {**info, "created": datetime.now(UTC).isoformat(timespec="seconds")}
    (root / "meta.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


def _read_meta(root: Path) -> dict:
    path = root / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {"source": "unknown"}


def cmd_synth(args) -> None:
    from timefence.data.synthetic import synthetic_store

    store = synthetic_store(
        n_stocks=args.stocks, n_days=args.days, reversal=args.reversal, seed=args.seed
    )
    root = Path(args.out)
    store.save(root)
    _write_meta(root, {"source": "synthetic", "stocks": args.stocks, "days": args.days})
    print(
        f"synthetic market saved to {root}: {len(store.calendar)} days, {len(store.stocks)} stocks"
    )


def cmd_download(args) -> None:
    from timefence.data.ingest import download

    root = Path(args.out)
    store = download(root, start=args.start, end=args.end, workers=args.workers, limit=args.limit)
    _write_meta(root, {"source": "eastmoney via akshare", "start": args.start, "end": args.end})
    cal = store.calendar
    print(
        f"saved {len(store.bars):,} bars for {len(store.stocks)} stocks, {cal[0].date()} to {cal[-1].date()}"
    )


def cmd_run(args) -> None:
    from timefence.experiments.runner import RunConfig, run_experiment

    root = Path(args.data)
    store = MarketStore.load(root)
    cfg = RunConfig(
        method=args.method, offline=args.offline, rounds=args.rounds,
        proposals_per_round=args.proposals, random_trials=args.trials, seed=args.seed,
        top_n=args.top_n, top_k=args.top_k, provider=args.provider, model=args.model,
        thinking=args.thinking, effort=args.effort, base_url=args.base_url,
        api_key_env=args.api_key_env, max_calls=args.max_calls, holdout=args.holdout,
        holdout_start=args.holdout_start,
    )  # fmt: skip
    summary = run_experiment(store, cfg, args.out, _read_meta(root))
    bt = summary["backtest"] or {}
    kept = sum(f["kept"] for f in summary["factors"])
    print(f"accepted {summary['mining']['counts']['accepted']}, kept after validation {kept}")
    if bt:
        print(
            f"test window: annual excess {_fmt(bt.get('excess_annual_return'), '+.1%')}, "
            f"IR {_fmt(bt.get('information_ratio'), '+.2f')}"
        )
    print(f"report: {Path(args.out) / 'report.md'}")


def cmd_compare(args) -> None:
    from timefence.experiments.report import write_comparison

    print(write_comparison(args.runs, args.out))


def cmd_audit(args) -> None:
    """Recompute random factors with the future deleted; any difference is a leak."""
    from timefence.data.universe import UniverseConfig
    from timefence.experiments.baselines import HANDCRAFTED
    from timefence.factors.evaluate import ResearchContext
    from timefence.factors.generate import random_expression

    store = MarketStore.load(args.data)
    ucfg = UniverseConfig(top_n=args.top_n)
    rng = random.Random(args.seed)
    expressions = list(HANDCRAFTED.values()) + [random_expression(rng) for _ in range(args.n)]
    full = ResearchContext(store, ucfg.build(store))
    cal = store.calendar
    cuts = sorted(rng.sample(range(len(cal) // 3, len(cal) - 10), args.cuts))
    failures = 0
    for pos in cuts:
        short = store.slice_until(cal[pos])
        part = ResearchContext(short, ucfg.build(short))
        for expr in expressions:
            a = full.compute(expr)[1].loc[: cal[pos], part.codes]
            b = part.compute(expr)[1]
            same = ((a - b).abs() <= 1e-9 * (1 + b.abs())) | (a.isna() & b.isna())
            if not same.all().all():
                failures += 1
                print(f"LEAK? {expr} differs when data after {cal[pos].date()} is removed")
    total = len(cuts) * len(expressions)
    print(f"{total - failures}/{total} checks identical with the future deleted")
    sys.exit(1 if failures else 0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="timefence", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="get market data").add_subparsers(
        dest="data_command", required=True
    )
    p = data.add_parser("synth", help="generate a synthetic market for demos")
    p.add_argument("--out", default="data/synth")
    p.add_argument("--stocks", type=int, default=300)
    p.add_argument("--days", type=int, default=1500)
    p.add_argument("--reversal", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_synth)
    p = data.add_parser("download", help="download A-share daily bars (needs eastmoney access)")
    p.add_argument("--out", default="data/ashare")
    p.add_argument("--start", default="20160101")
    p.add_argument("--end", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--limit", type=int, default=None, help="only the first N stocks (for a quick try)"
    )
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("run", help="mine, validate and test one method")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="run folder, e.g. runs/agent")
    p.add_argument("--method", choices=["agent", "random", "handcrafted"], default="agent")
    p.add_argument(
        "--offline", action="store_true", help="agent without a model: template stand-in"
    )
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--proposals", type=int, default=4)
    p.add_argument("--trials", type=int, default=40, help="random baseline: factors to score")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-n", type=int, default=500, help="universe size")
    p.add_argument("--top-k", type=int, default=50, help="stocks held")
    p.add_argument(
        "--provider",
        choices=list(PROVIDERS),
        default="claude",
        help="model provider; it also sets where the holdout starts (default: claude)",
    )
    p.add_argument("--model", help="default: the provider's model")
    p.add_argument(
        "--thinking",
        choices=["disabled", "enabled", "omit"],
        help="DeepSeek thinking mode (default: disabled); 'omit' leaves the field out",
    )
    p.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="K2 Horizon and Claude: how hard the model reasons (default: high)",
    )
    p.add_argument("--base-url", help="default: the provider's endpoint, or $K2_BASE_URL for K2")
    p.add_argument("--api-key-env", help="variable holding the key (default: the provider's)")
    p.add_argument("--max-calls", type=int, default=300)
    p.add_argument("--holdout", action="store_true", help="also run the final check on later data")
    p.add_argument(
        "--holdout-start",
        help="first holdout day (default: the day the provider's model stops knowing the market)",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="compare finished runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="runs/comparison.md")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser(
        "audit", help="check that no factor value changes when the future is deleted"
    )
    p.add_argument("--data", required=True)
    p.add_argument(
        "--n", type=int, default=20, help="random expressions on top of the handcrafted ones"
    )
    p.add_argument("--cuts", type=int, default=3)
    p.add_argument("--top-n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_audit)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    pd.set_option("display.width", 120)
    _load_dotenv()
    args.func(args)


if __name__ == "__main__":
    main()
