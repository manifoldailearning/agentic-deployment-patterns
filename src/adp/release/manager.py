"""Pattern 16: release management for prompts and models.

A *variant* is everything that changes behaviour: prompt
version, model and settings. Variants are immutable and
content-addressed. Traffic is split deterministically by
request id: stable, a small canary slice, and an optional
shadow that runs on a copy of traffic but is never served.
Health checks on the canary trigger automatic rollback.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Variant:
    prompt_version: str
    model: str
    temperature: float = 0.0

    @property
    def id(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]


def bucket(request_id: str, salt: str = "release") -> float:
    """Stable value in [0, 100) for a request id."""
    h = hashlib.sha256(f"{salt}:{request_id}".encode())
    return int(h.hexdigest()[:8], 16) % 10_000 / 100


@dataclass
class Stats:
    requests: int = 0
    errors: int = 0
    eval_pass: int = 0
    eval_total: int = 0

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0

    @property
    def pass_rate(self) -> float:
        return (self.eval_pass / self.eval_total
                if self.eval_total else 1.0)


@dataclass(frozen=True)
class RolloutPolicy:
    min_requests: int = 50
    max_error_rate_delta: float = 0.02
    max_pass_rate_drop: float = 0.03


@dataclass
class ReleaseState:
    stable: Variant
    canary: Variant | None = None
    canary_percent: float = 0.0
    shadow: Variant | None = None
    history: list[str] = field(default_factory=list)


class ReleaseManager:
    def __init__(self, stable: Variant,
                 policy: RolloutPolicy | None = None) -> None:
        self.state = ReleaseState(stable)
        self.policy = policy or RolloutPolicy()
        self.stats: dict[str, Stats] = {stable.id: Stats()}
        self.shadow_diffs: list[tuple[str, str, str]] = []

    # ----- configuration -------------------------------------
    def start_canary(self, v: Variant, percent: float) -> None:
        if not 0 < percent <= 50:
            raise ValueError("canary percent must be in (0, 50]")
        self.state.canary, self.state.canary_percent = v, percent
        self.stats.setdefault(v.id, Stats())
        self._log(f"canary {v.id} at {percent}%")

    def set_shadow(self, v: Variant | None) -> None:
        self.state.shadow = v
        self._log(f"shadow {v.id if v else 'off'}")

    def promote(self) -> None:
        if self.state.canary is None:
            raise ValueError("no canary to promote")
        self.state.stable = self.state.canary
        self._clear_canary(f"promoted {self.state.stable.id}")

    def rollback(self, reason: str) -> None:
        if self.state.canary is not None:
            self._clear_canary(
                f"rolled back {self.state.canary.id}: {reason}")

    # ----- serving -------------------------------------------
    def choose(self, request_id: str) -> Variant:
        s = self.state
        if s.canary and bucket(request_id) < s.canary_percent:
            return s.canary
        return s.stable

    def serve(self, request_id: str,
              run: Callable[[Variant], str]) -> tuple[str, str]:
        v = self.choose(request_id)
        st = self.stats[v.id]
        st.requests += 1
        try:
            out = run(v)
        except Exception:
            st.errors += 1
            self.check_canary()
            raise
        sh = self.state.shadow
        if sh is not None and sh.id != v.id:
            try:
                shadow_out = run(sh)
            except Exception as exc:
                shadow_out = f"ERROR {type(exc).__name__}"
            if shadow_out != out:
                self.shadow_diffs.append(
                    (request_id, out, shadow_out))
        self.check_canary()
        return out, v.id

    def record_eval(self, variant_id: str, passed: bool) -> None:
        st = self.stats.setdefault(variant_id, Stats())
        st.eval_total += 1
        st.eval_pass += int(passed)

    def check_canary(self) -> str | None:
        c = self.state.canary
        if c is None:
            return None
        cs = self.stats[c.id]
        ss = self.stats[self.state.stable.id]
        p = self.policy
        if cs.requests < p.min_requests:
            return None
        if cs.error_rate > ss.error_rate + p.max_error_rate_delta:
            reason = (f"error rate {cs.error_rate:.1%} vs "
                      f"{ss.error_rate:.1%}")
        elif cs.pass_rate < ss.pass_rate - p.max_pass_rate_drop:
            reason = (f"eval pass {cs.pass_rate:.1%} vs "
                      f"{ss.pass_rate:.1%}")
        else:
            return None
        self.rollback(reason)
        return reason

    def _clear_canary(self, msg: str) -> None:
        self.state.canary, self.state.canary_percent = None, 0.0
        self._log(msg)

    def _log(self, msg: str) -> None:
        self.state.history.append(msg)
