"""The research loop as a LangGraph state machine.

    plan -> propose -> judge -> review -> (next round: plan | done)
                ^          |
                +- retry --+   when no proposal was usable

plan, propose and review are model calls (the planner, researcher and reviewer roles).
judge is plain code: it parses, de-duplicates, scores on the training window and applies
the acceptance rules, so no prompt can talk a factor into the library.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from timefence.agents import prompts
from timefence.agents.llm import LLM, parse_json
from timefence.agents.memory import ResearchMemory
from timefence.agents.tools import FactorLab
from timefence.core.errors import BudgetExceeded
from timefence.factors import dsl

log = logging.getLogger(__name__)


class RoundState(TypedDict, total=False):
    round: int
    theme: str
    brief: str
    proposals: list[dict]
    feedback: str  # why the last proposals were unusable, for a retry
    retries: int
    results: str  # this round's verdicts as text, for the reviewer


class ResearchAgent:
    def __init__(
        self,
        llm: LLM,
        lab: FactorLab,
        memory: ResearchMemory,
        run_id: str,
        *,
        rounds: int = 10,
        proposals_per_round: int = 4,
        max_retries: int = 1,
        horizon: int = 5,
        transcript: str | Path | None = None,
    ):
        self.llm = llm
        self.lab = lab
        self.memory = memory
        self.run_id = run_id
        self.rounds = rounds
        self.n = proposals_per_round
        self.max_retries = max_retries
        self.horizon = horizon
        self.transcript = Path(transcript) if transcript else None
        self.stop_reason = "finished all rounds"
        self.graph = self._build()

    # ---- graph --------------------------------------------------------------------------

    def _build(self):
        g = StateGraph(RoundState)
        g.add_node("plan", self.plan)
        g.add_node("propose", self.propose)
        g.add_node("judge", self.judge)
        g.add_node("review", self.review)
        g.add_edge(START, "plan")
        g.add_edge("plan", "propose")
        g.add_edge("propose", "judge")
        g.add_conditional_edges(
            "judge", self._after_judge, {"retry": "propose", "review": "review"}
        )
        g.add_conditional_edges("review", self._after_review, {"next": "plan", "stop": END})
        return g.compile()

    def run(self) -> dict:
        try:
            self.graph.invoke(
                {"round": 1, "retries": 0},
                {"recursion_limit": self.rounds * (6 + 2 * self.max_retries) + 10},
            )
        except BudgetExceeded as exc:
            self.stop_reason = f"stopped early: {exc}"
            log.warning(self.stop_reason)
        return {
            "counts": self.memory.counts(self.run_id),
            "accepted": list(self.lab.library),
            "stop_reason": self.stop_reason,
            "model_calls": self.llm.budget.calls,
            "tokens": self.llm.budget.tokens,
        }

    def _after_judge(self, state: RoundState) -> str:
        return (
            "retry" if state.get("feedback") and state["retries"] <= self.max_retries else "review"
        )

    def _after_review(self, state: RoundState) -> str:
        return "next" if state["round"] <= self.rounds else "stop"

    # ---- nodes --------------------------------------------------------------------------

    def plan(self, state: RoundState) -> RoundState:
        system = prompts.PLANNER.format(themes="\n".join(f"- {t}" for t in prompts.THEMES))
        user = prompts.planner_user(state["round"], self.rounds, self._digest())
        reply = self._ask("planner", state["round"], system, user, temperature=0.7)
        fallback = prompts.THEMES[(state["round"] - 1) % len(prompts.THEMES)]
        theme = str(reply.get("theme") or fallback)
        brief = str(reply.get("brief") or f"Explore {theme}.")
        return {"theme": theme, "brief": brief, "retries": 0, "feedback": ""}

    def propose(self, state: RoundState) -> RoundState:
        system = prompts.RESEARCHER.format(
            horizon=self.horizon,
            language=dsl.describe_language(),
            window=str(self.lab.window),
            rules=self.lab.describe_rules(),
        )
        user = prompts.researcher_user(
            state["theme"], state["brief"], self._digest(), self.n, state.get("feedback", "")
        )
        reply = self._ask("researcher", state["round"], system, user, temperature=0.8)
        proposals = [
            p for p in reply.get("proposals", []) if isinstance(p, dict) and p.get("expression")
        ]
        return {"proposals": proposals[: self.n], "retries": state.get("retries", 0) + 1}

    def judge(self, state: RoundState) -> RoundState:
        verdicts = []
        for p in state.get("proposals", []):
            v = self.lab.judge(
                str(p["expression"]), lambda c: self.memory.status_of(self.run_id, c)
            )
            self.memory.add_trial(
                self.run_id, state["round"], v.expression, v.status,
                canonical=v.canonical, theme=state.get("theme"), hypothesis=p.get("hypothesis"),
                reason=v.reason, metrics=v.report.to_dict() if v.report else None,
            )  # fmt: skip
            verdicts.append(v)
        results = "\n\n".join(v.describe() for v in verdicts) or "(no proposals)"
        scored = [v for v in verdicts if v.status in ("accepted", "rejected")]
        feedback = "" if scored else results
        return {"results": results, "feedback": feedback}

    def review(self, state: RoundState) -> RoundState:
        user = prompts.reviewer_user(state.get("theme", ""), state.get("results", ""))
        reply = self._ask("reviewer", state["round"], prompts.REVIEWER, user, temperature=0.3)
        lessons = [str(x) for x in reply.get("lessons", []) if str(x).strip()][:3]
        if reply.get("next"):
            lessons.append(f"Next: {reply['next']}")
        for text in lessons:
            self.memory.add_note(self.run_id, state["round"], "reviewer", text)
        return {"round": state["round"] + 1, "proposals": [], "results": ""}

    # ---- helpers ------------------------------------------------------------------------

    def _digest(self) -> str:
        return self.memory.digest(self.run_id)

    def _ask(self, role: str, round_: int, system: str, user: str, temperature: float) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        reply = self.llm.complete(messages, temperature=temperature)
        try:
            parsed = parse_json(reply.text)
        except ValueError:
            log.warning("%s reply was not a JSON object: %.200s", role, reply.text)
            parsed = {}
        if self.transcript:
            self.transcript.parent.mkdir(parents=True, exist_ok=True)
            with self.transcript.open("a", encoding="utf-8") as f:
                record = {"round": round_, "role": role, "messages": messages, "reply": reply.text,
                          "cached": reply.cached, "tokens": reply.prompt_tokens + reply.completion_tokens}  # fmt: skip
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return parsed
