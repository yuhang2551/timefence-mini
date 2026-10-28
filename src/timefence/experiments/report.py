"""Markdown reports, in Chinese, for one run and for a comparison of runs."""

from __future__ import annotations

import json
from pathlib import Path

METHOD_NAMES = {"agent": "多智能体挖掘", "random": "随机表达式基线", "handcrafted": "经典因子基线"}


def _pct(x, digits=1, signed=True) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def _num(x, digits=2) -> str:
    return "n/a" if x is None else f"{x:+.{digits}f}"


def _warnings(summary: dict) -> list[str]:
    out = []
    source = summary.get("data", {}).get("source", "unknown")
    if source == "synthetic":
        out.append(
            "**数据是程序生成的模拟行情**，只用来检查流程是否跑通。下面所有数字都不是研究结果。"
        )
    model = summary.get("mining", {}).get("model", "")
    if "not a language model" in model:
        out.append("**挖掘用的是离线模板替身，不是大模型**。它只证明流程能跑，不代表大模型的能力。")
    b = summary.get("holdout_boundary") or {}
    if b.get("before_bound"):
        out.append(
            f"**留出期从 {b['start']} 开始，早于 {b['model']} 的知识边界 {b['knowledge_bound']}**，"
            "模型可能见过其中一部分行情。"
        )
    return out


def _boundary(b: dict) -> str:
    reason = {
        "cutoff": f"{b['model']} 公布的知识截止之后",
        "release": f"{b['model']} 没有公布知识截止，按发布日算",
        "manual": "运行时指定",
    }[b["basis"]]
    return f"{b['start']} 起（{reason}），训练、验证、测试集都在它之前"


def write_report(summary: dict, path: str | Path) -> None:
    cfg, mining = summary["config"], summary["mining"]
    boundary = summary.get("holdout_boundary")
    lines = [f"# 运行报告：{summary['run_id']}", ""]
    for w in _warnings(summary):
        lines += [f"> ⚠️ {w}", ""]

    lines += [
        "## 运行信息",
        "",
        f"- 方法：{METHOD_NAMES.get(cfg['method'], cfg['method'])}"
        + (f"（模型：{mining.get('model')}）" if mining.get("model") else ""),
        f"- 数据：{summary.get('data', {}).get('source', 'unknown')}",
        f"- 训练集 {summary['split']['train']}，验证集 {summary['split']['valid']}，"
        f"测试集 {summary['split']['test']}",
        *([f"- 留出期：{_boundary(boundary)}"] if boundary else []),
        f"- 股票池：每天成交额前 {cfg['top_n']} 只（上市满 {cfg['min_listed_days']} 天，剔除北交所）",
        f"- 预测周期：{cfg['horizon']} 个交易日（收盘出信号，次日开盘买入）",
        "",
        "## 挖掘过程（只用训练集）",
        "",
    ]
    c = mining["counts"]
    lines.append(
        f"共尝试 {sum(c.values())} 个表达式：通过 {c['accepted']}，未达标 {c['rejected']}，"
        f"语法不合法 {c['invalid']}，重复 {c['duplicate']}。"
    )
    if "model_calls" in mining:
        lines.append(
            f"模型调用 {mining['model_calls']} 次，约 {mining['tokens']:,} tokens。{mining['stop_reason']}。"
        )
    lines.append("")

    factors = summary["factors"]
    if factors:
        lines += [
            "## 因子库",
            "",
            "方向由训练集决定。验证集上方向一致且 |RankICIR| 达标的因子进入组合。",
            "",
            "| 因子表达式 | 方向 | 训练 RankIC | 训练 ICIR | 验证 RankIC | 验证 ICIR | 进组合 | 测试 RankIC | 测试 ICIR |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for f in factors:
            t, v, s = f["train"], f["valid"], f.get("test") or {}
            lines.append(
                f"| `{f['expression']}` | {f['direction']:+d} | {_num(t['rank_ic'], 4)} | "
                f"{_num(t['rank_icir'])} | {_num(v['rank_ic'], 4)} | {_num(v['rank_icir'])} | "
                f"{'是' if f['kept'] else '否'} | {_num(s.get('rank_ic'), 4)} | {_num(s.get('rank_icir'))} |"
            )
        lines.append("")

    bt = summary.get("backtest")
    lines += ["## 组合回测（测试集，样本外，只跑一次）", ""]
    if bt:
        lines += [
            f"持有得分最高的 {cfg['top_k']} 只股票等权，每 {cfg['rebalance_every']} 个交易日调仓；"
            "A 股规则：T+1、涨跌停买卖不了、停牌、100 股一手、佣金最低 5 元、印花税 2023-08-28 减半。"
            "基准是同一股票池每日等权（不计成本）。",
            "",
            "| 指标 | 策略 | 基准 |",
            "|---|---|---|",
            f"| 年化收益 | {_pct(bt['annual_return'])} | {_pct(bt.get('benchmark_annual_return'))} |",
            f"| 年化波动 | {_pct(bt['annual_vol'], signed=False)} | |",
            f"| 夏普比率 | {_num(bt['sharpe'])} | |",
            f"| 最大回撤 | {_pct(bt['max_drawdown'])} | |",
            f"| 年化超额收益 | {_pct(bt.get('excess_annual_return'))} | |",
            f"| 信息比率 | {_num(bt.get('information_ratio'))} | |",
            f"| 超额最大回撤 | {_pct(bt.get('excess_max_drawdown'))} | |",
            f"| 月度胜率 | {_pct(bt.get('monthly_win_rate'), 0, signed=False)} | |",
            f"| 年换手（单边） | {bt['annual_turnover']:.1f} 倍 | |",
            f"| 年交易成本 | {_pct(bt['annual_cost'], 2, signed=False)} | |",
            f"| 被规则挡住的订单 | {int(bt['blocked_orders'])} | |",
            "",
        ]
    else:
        lines += ["没有因子通过验证集，所以没有组合可以回测。", ""]

    if summary.get("holdout"):
        h = summary["holdout"]["backtest"]
        lines += [
            "## 最终留出期",
            "",
            f"{summary['holdout']['window']}：年化收益 {_pct(h['annual_return'])}，"
            f"年化超额 {_pct(h.get('excess_annual_return'))}，信息比率 {_num(h.get('information_ratio'))}。",
            "",
        ]
    elif cfg.get("holdout"):
        lines += [
            "## 最终留出期",
            "",
            "没有跑：留出期起点之后不到 60 个交易日，或者没有因子进组合。",
            "",
        ]

    lines += [
        "## 怎么读这份报告",
        "",
        "- 训练集的数字是挖掘时看到的，天然偏乐观；验证集和测试集才说明因子有没有用。",
        "- 测试集只在所有选择都定下来之后跑一次。看了测试结果再回去改因子，测试集就不再是样本外。",
        "- 测试集挡住的是反复试错带来的过拟合，但大模型训练时可能见过测试集那几年的行情，“记得”什么因子好用。"
        "留出期从模型知识截止之后开始（没有公布截止日的模型按发布日算），才能排除这一点。",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_comparison(run_dirs: list[str | Path], path: str | Path) -> str:
    rows = [
        "| 运行 | 方法 | 打分次数 | 训练通过 | 验证保留 | 保留因子测试 RankIC 均值 | 年化超额 | 信息比率 | 年换手 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    splits = set()  # the table compares test windows, so the splits have to match
    mismatch = ["> ⚠️ 这几次运行的训练、验证、测试划分不一样，数字不能直接比较。", ""]
    for d in run_dirs:
        s = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))
        splits.add(tuple(s["split"].values()))
        kept = [f for f in s["factors"] if f["kept"]]
        tests = [f["test"]["rank_ic"] * f["direction"] for f in kept if f.get("test")]
        mean_ic = sum(tests) / len(tests) if tests else None
        bt = s.get("backtest") or {}
        rows.append(
            f"| {s['run_id']} | {METHOD_NAMES.get(s['config']['method'], s['config']['method'])} | "
            f"{s['mining']['scored']} | {s['mining']['counts']['accepted']} | {len(kept)} | "
            f"{_num(mean_ic, 4)} | {_pct(bt.get('excess_annual_return'))} | "
            f"{_num(bt.get('information_ratio'))} | "
            f"{'n/a' if not bt else format(bt['annual_turnover'], '.1f') + ' 倍'} |"
        )
    text = "\n".join(
        [
            "# 方法对比",
            "",
            *(mismatch if len(splits) > 1 else []),
            "测试 RankIC 已按训练集方向调整，正数表示方向在测试集上依然成立。",
            "",
            *rows,
            "",
        ]
    )
    Path(path).write_text(text, encoding="utf-8")
    return text
