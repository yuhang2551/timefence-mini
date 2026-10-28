[中文](CHANGELOG.md) | English

# Changelog

## Next

1. Run on real data: the `audit` self-check, the random and handcrafted baselines, the agent with K2 Horizon, then `compare` for the comparison table, and fill in the README results.
2. Once the whole pipeline runs, switch to Claude (`--provider claude`) and rerun all three, so they share the same holdout start.

## 2026-10-28: switchable model provider; the holdout follows the model

- `--provider k2 | claude | deepseek` picks the model; K2 Horizon (MBZUAI's Institute of Foundation Models) is the default. K2 Horizon and DeepSeek use the OpenAI-compatible protocol; K2 Horizon reads its API address from `K2_BASE_URL`, takes `--effort` for reasoning depth and uses the sampling its model card recommends. Claude uses the official Anthropic SDK (streamed replies, `--effort` for thinking depth, a cached system prompt, and a server-side retry on another model after a policy refusal).
- By default the holdout starts at the chosen model's knowledge bound (the day after a published cutoff, otherwise the release date) instead of a fixed 2026-01-01. A model that is not a registered default needs an explicit `--holdout-start`. Reports say where the holdout starts and why; `compare` warns when runs have different splits.
- `.env.example` lists every variable the providers use, with empty values.
- Replies are parsed after any `</think>` reasoning, for servers that do not split the reasoning from the answer.
- README commands no longer carry trailing `#` comments, which zsh would run as commands.
- 141 tests pass.

## 2026-10-17: title and DeepSeek fix

- The Chinese title changed from 看不到未来 to 杜绝未来函数, because the old phrase also reads as "no future".
- DeepSeek retired `deepseek-chat` and other old model names on 2026-07-24, and its new models think by default. The default model became `deepseek-flash` with thinking off.
- Neither was released at the time; both ship together with the 2026-10-28 changes.

## 2026-10-13: first version

- The one-week version: the time fence and truncation-equivalence tests, a factor language with 21 operators, an A-share backtest, LangGraph mining with three roles, random and handcrafted baselines, and Chinese reports. 133 tests, CI green.
