[中文](CHANGELOG.md) | English

# Changelog

## Next

1. Run on real data somewhere that can reach Eastmoney: the `audit` self-check, the random and handcrafted baselines, the agent with Claude Haiku 5.5, then `compare` for the comparison table, and fill in the README results.
2. For a stronger model, run the agent again with `--model claude-opus-5-5`: same knowledge cutoff, same holdout, no need to rerun the baselines.

## 2026-10-28 (second update): Claude by default, starting with the cheapest model

- The default provider is now Claude instead of K2 Horizon, with the cheapest model, Claude Haiku 5.5 ($0.10 input and $0.50 output per million tokens for prompts up to 100K tokens). Claude Opus 5.5 has the same June 2026 knowledge cutoff and is registered too, so `--model claude-opus-5-5` needs no separate holdout date.
- Claude Haiku 5.5 has no server-side retry on another model, so its requests no longer ask for one.
- K2 Horizon goes through IFM's own API, `https://api.ifm.ai/v1`, by default and needs only `K2_API_KEY`; `K2_BASE_URL` is for other hosts.
- The first agent run with a real model: Claude Haiku 5.5 on simulated market data finished all 10 rounds in 31 calls (about 100K tokens, about $0.03). The planner, researcher and reviewer replies all parsed; of 40 expressions tried, 11 passed the training checks. Once the researcher answered in prose instead of JSON, and the loop treated it as an unusable reply and asked again without stopping. Numbers on simulated data are not research results; the run only shows that the whole pipeline works.
- The model reply cache, `.cache/`, is now in `.gitignore`.
- 143 tests pass.

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
