"""Map natural language to typed tool calls via one Jev fan-out request."""

from __future__ import annotations

import inspect
import json
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, get_args, get_origin, get_type_hints

from typesafe_sdk import Choice, Noul, TypeSafeClient

ROUTE = "__tool__"
STATED_NOUL_THRESHOLD = 0.5

SPEC_PATH = Path(__file__).with_name("spec.json")


def _unwrap_optional(ann: Any) -> Any:
    origin = get_origin(ann)
    if origin is types.UnionType or str(origin) == "typing.Union":
        non_none = [a for a in get_args(ann) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return ann


def closed_sets(fn: Callable[..., Any]) -> dict[str, tuple[str, list[str] | None]]:
    """Return arg -> (choice|set|flag, options) from Literal / bool annotations."""
    from typing import Literal as LiteralType

    hints = get_type_hints(fn)
    shapes: dict[str, tuple[str, list[str] | None]] = {}
    for name in inspect.signature(fn).parameters:
        if name not in hints:
            continue
        ann = _unwrap_optional(hints[name])
        origin = get_origin(ann)
        if ann is bool:
            shapes[name] = ("flag", None)
            continue
        if origin is list:
            inner = _unwrap_optional(get_args(ann)[0]) if get_args(ann) else None
            if get_origin(inner) is LiteralType:
                shapes[name] = ("set", [str(x) for x in get_args(inner)])
            continue
        if origin is LiteralType:
            shapes[name] = ("choice", [str(x) for x in get_args(ann)])
    return shapes


@dataclass
class ArgResult:
    name: str
    value: Any = None
    probability: float = 1.0
    distribution: dict[str, float] = field(default_factory=dict)
    omitted: bool = False


FREE_TEXT_PARAMS = frozenset({"query", "expression"})


def _free_text_value(name: str, utterance: str) -> str:
    from agent.actions import extract_math_expression, extract_search_query

    if name == "query":
        return extract_search_query(utterance)
    if name == "expression":
        return extract_math_expression(utterance)
    return utterance


@dataclass
class Call:
    name: str
    fn: Callable[..., Any]
    arguments: dict[str, ArgResult]
    tool_probability: float
    confidence: float
    utterance: str = ""
    elapsed_ms: float = 0.0

    def __str__(self) -> str:
        parts = []
        for name, arg in self.arguments.items():
            if arg.omitted:
                continue
            parts.append(f"{name}={arg.value!r}")
        filled = {a for a, v in self.arguments.items() if not v.omitted}
        sig = inspect.signature(self.fn)
        for name in FREE_TEXT_PARAMS:
            if name in sig.parameters and name not in filled:
                parts.append(f"{name}={_free_text_value(name, self.utterance)!r}")
        return f"{self.name}({', '.join(parts)})"

    def run(self) -> Any:
        kwargs = {n: a.value for n, a in self.arguments.items() if not a.omitted}
        sig = inspect.signature(self.fn)
        for name in FREE_TEXT_PARAMS:
            if name in sig.parameters and name not in kwargs:
                kwargs[name] = _free_text_value(name, self.utterance)
        return self.fn(**kwargs)

    def weakest(self) -> ArgResult | None:
        if not self.arguments:
            return None
        return min(self.arguments.values(), key=lambda a: a.probability)


class Dispatcher:
    def __init__(
        self,
        spec: dict[str, Any],
        tools: dict[str, Callable[..., Any]],
        client: TypeSafeClient,
    ) -> None:
        self.spec = spec
        self.tools = tools
        self.client = client
        self.shapes = {name: closed_sets(fn) for name, fn in tools.items()}
        self.questions = self._build_questions()

    @classmethod
    def from_default(cls, tools: dict[str, Callable[..., Any]], client: TypeSafeClient) -> Dispatcher:
        spec = json.loads(SPEC_PATH.read_text())
        return cls(spec, tools, client)

    def _build_questions(self) -> dict[str, Any]:
        functions = self.spec["functions"]
        route_q = self.spec.get("route", {}).get(
            "question",
            "What is the user asking the local Mac assistant to do?",
        )
        questions: dict[str, Any] = {
            ROUTE: Choice(
                instructions=route_q,
                criteria={
                    name: functions[name]["description"]
                    for name in self.tools
                    if name in functions
                },
            )
        }

        for fname, fn in self.tools.items():
            fspec = functions.get(fname, {})
            arg_specs = fspec.get("arguments", {})
            for aname, (shape, options) in self.shapes[fname].items():
                aspec = arg_specs.get(aname, {})
                qid = f"{fname}.{aname}"
                if shape == "flag":
                    questions[qid] = Noul(
                        instructions=aspec.get(
                            "question",
                            f"Should {aname} be enabled for this request?",
                        )
                    )
                    if "stated" in aspec:
                        questions[f"{qid}?"] = Noul(instructions=aspec["stated"])
                elif shape == "choice":
                    option_text = aspec.get("options") or {o: o for o in (options or [])}
                    questions[qid] = Choice(
                        instructions=aspec.get(
                            "question",
                            f"Which value of {aname} does the user want?",
                        ),
                        criteria=option_text,
                    )
                    if "stated" in aspec:
                        questions[f"{qid}?"] = Noul(instructions=aspec["stated"])
                elif shape == "set":
                    template = aspec.get(
                        "question",
                        "Does the user want {} included?",
                    )
                    for member in options or []:
                        questions[f"{qid}.{member}"] = Noul(
                            instructions=template.replace("{}", member)
                        )
        return questions

    def __call__(self, utterance: str) -> Call:
        import time

        t0 = time.perf_counter()
        response = self.client.system_one(state=utterance, questions=self.questions)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        route = response.answers[ROUTE]
        tool_name = route.choice
        tool_p = float(route.probabilities.get(tool_name, route.confidence))
        fn = self.tools[tool_name]

        args: dict[str, ArgResult] = {}
        probs: list[float] = [float(route.confidence)]

        for aname, (shape, options) in self.shapes[tool_name].items():
            qid = f"{tool_name}.{aname}"
            stated_id = f"{qid}?"
            omitted = False
            if stated_id in response.answers:
                stated = float(response.answers[stated_id].noul)
                if stated < STATED_NOUL_THRESHOLD:
                    omitted = True
                    args[aname] = ArgResult(
                        name=aname,
                        omitted=True,
                        probability=1.0 - stated,
                    )
                    probs.append(1.0 - stated)
                    continue

            if shape == "flag":
                ans = response.answers[qid]
                value = float(ans.noul) >= 0.5
                p = float(ans.noul) if value else 1.0 - float(ans.noul)
                args[aname] = ArgResult(name=aname, value=value, probability=p)
                probs.append(p)
            elif shape == "choice":
                ans = response.answers[qid]
                value = ans.choice
                p = float(ans.probabilities.get(value, ans.confidence))
                args[aname] = ArgResult(
                    name=aname,
                    value=value,
                    probability=p,
                    distribution={k: float(v) for k, v in ans.probabilities.items()},
                )
                probs.append(float(ans.confidence))
            elif shape == "set":
                selected: list[str] = []
                member_ps: list[float] = []
                for member in options or []:
                    ans = response.answers[f"{qid}.{member}"]
                    n = float(ans.noul)
                    if n >= 0.5:
                        selected.append(member)
                        member_ps.append(n)
                    else:
                        member_ps.append(1.0 - n)
                p = min(member_ps) if member_ps else 1.0
                args[aname] = ArgResult(name=aname, value=selected, probability=p)
                probs.append(p)

        confidence = min(probs) if probs else 0.0
        return Call(
            name=tool_name,
            fn=fn,
            arguments=args,
            tool_probability=tool_p,
            confidence=confidence,
            utterance=utterance,
            elapsed_ms=elapsed_ms,
        )
