"""Confidence gate before executing a Call."""

from __future__ import annotations

from dataclasses import dataclass

from agent.dispatch import Call

DEFAULT_THRESHOLD = 0.7


@dataclass
class GateResult:
    allowed: bool
    call: Call
    reason: str


def gate(call: Call, threshold: float = DEFAULT_THRESHOLD) -> GateResult:
    if call.name == "clarify":
        return GateResult(True, call, "clarify")
    if call.confidence < threshold:
        return GateResult(
            False,
            call,
            f"confidence {call.confidence:.2f} below threshold {threshold:.2f}",
        )
    return GateResult(True, call, "ok")
