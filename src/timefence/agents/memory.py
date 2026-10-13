"""Research memory in SQLite: every factor tried, why it passed or failed, and lessons.

The agents read a short digest of it each round, so they build on what worked and do not
repeat what failed. Everything is scoped by `run_id`, so several runs can share one file.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    theme TEXT,
    hypothesis TEXT,
    expression TEXT NOT NULL,      -- as proposed
    canonical TEXT,                -- normalised form; NULL if it did not parse
    status TEXT NOT NULL,          -- invalid | duplicate | rejected | accepted
    reason TEXT,                   -- parse error or the rules it failed
    metrics TEXT,                  -- JSON factor report on the training window
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS trials_run_canonical ON trials(run_id, canonical);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    author TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

STATUSES = ("invalid", "duplicate", "rejected", "accepted")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ResearchMemory:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    # ---- writing ------------------------------------------------------------------------

    def add_trial(
        self,
        run_id: str,
        round_: int,
        expression: str,
        status: str,
        *,
        canonical: str | None = None,
        theme: str | None = None,
        hypothesis: str | None = None,
        reason: str | None = None,
        metrics: dict | None = None,
    ) -> int:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        cur = self._db.execute(
            "INSERT INTO trials (run_id, round, theme, hypothesis, expression, canonical, status,"
            " reason, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, round_, theme, hypothesis, expression, canonical, status, reason,
             json.dumps(metrics) if metrics is not None else None, _now()),
        )  # fmt: skip
        self._db.commit()
        return int(cur.lastrowid)

    def add_note(self, run_id: str, round_: int, author: str, text: str) -> None:
        self._db.execute(
            "INSERT INTO notes (run_id, round, author, text, created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, round_, author, text, _now()),
        )
        self._db.commit()

    # ---- reading ------------------------------------------------------------------------

    def status_of(self, run_id: str, canonical: str) -> str | None:
        """How this factor fared earlier in the run, or None if it is new."""
        row = self._db.execute(
            "SELECT status FROM trials WHERE run_id = ? AND canonical = ? AND status != 'duplicate'"
            " ORDER BY id LIMIT 1",
            (run_id, canonical),
        ).fetchone()
        return row["status"] if row else None

    def trials(self, run_id: str, status: str | None = None) -> list[dict]:
        sql = "SELECT * FROM trials WHERE run_id = ?"
        args: tuple = (run_id,)
        if status:
            sql += " AND status = ?"
            args += (status,)
        rows = self._db.execute(sql + " ORDER BY id", args).fetchall()
        return [self._row(r) for r in rows]

    def notes(self, run_id: str, limit: int = 20) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM notes WHERE run_id = ? ORDER BY id DESC LIMIT ?", (run_id, limit)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def counts(self, run_id: str) -> dict[str, int]:
        rows = self._db.execute(
            "SELECT status, COUNT(*) AS n FROM trials WHERE run_id = ? GROUP BY status", (run_id,)
        ).fetchall()
        out = dict.fromkeys(STATUSES, 0)
        out.update({r["status"]: r["n"] for r in rows})
        return out

    def digest(self, run_id: str, n_best: int = 8, n_failed: int = 10, n_notes: int = 6) -> str:
        """A compact summary for prompts: best factors so far, recent failures, lessons."""
        lines = []
        accepted = self.trials(run_id, "accepted")
        accepted.sort(key=lambda t: -abs(t["metrics"]["rank_icir"]))
        if accepted:
            lines.append("Accepted factors (best first):")
            for t in accepted[:n_best]:
                m = t["metrics"]
                lines.append(
                    f"- {t['canonical']}  RankIC {m['rank_ic']:+.4f}, ICIR {m['rank_icir']:+.2f}, "
                    f"turnover {m['turnover']:.0%}"
                )
        else:
            lines.append("No accepted factors yet.")
        failed = [t for t in self.trials(run_id) if t["status"] in ("rejected", "invalid")]
        if failed:
            lines.append("Recent failures:")
            for t in failed[-n_failed:]:
                lines.append(f"- {t['canonical'] or t['expression']}: {t['status']}, {t['reason']}")
        notes = self.notes(run_id, n_notes)
        if notes:
            lines.append("Lessons so far:")
            lines.extend(f"- {n['text']}" for n in notes)
        c = self.counts(run_id)
        lines.append(
            f"Totals: {sum(c.values())} tried, {c['accepted']} accepted, {c['rejected']} rejected, "
            f"{c['invalid']} invalid, {c['duplicate']} duplicates."
        )
        return "\n".join(lines)

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        out = dict(row)
        out["metrics"] = json.loads(out["metrics"]) if out["metrics"] else None
        return out
