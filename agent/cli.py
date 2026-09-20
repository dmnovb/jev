"""CLI entrypoint: one-shot or REPL."""

from __future__ import annotations

import argparse
import sys

from agent.actions import TOOLS
from agent.client import make_client
from agent.dispatch import Dispatcher
from agent.gate import DEFAULT_THRESHOLD, gate


def handle(dispatcher: Dispatcher, utterance: str, threshold: float) -> int:
    call = dispatcher(utterance)
    decision = gate(call, threshold=threshold)
    print(f"  {call}   confidence {call.confidence:.2f}   {call.elapsed_ms:.0f}ms")
    if not decision.allowed:
        print(f"  refused: {decision.reason}")
        return 1
    result = call.run()
    if result:
        print(f"  -> {result}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="jev",
        description="Local Mac assistant powered by TypeSafe Jev",
    )
    parser.add_argument(
        "utterance",
        nargs="?",
        help='Command to run, e.g. "open notes". Omit for a REPL.',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum confidence to execute (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Decide only; do not execute",
    )
    args = parser.parse_args(argv)

    client = make_client()
    dispatcher = Dispatcher.from_default(TOOLS, client)

    if args.utterance:
        if args.dry_run:
            call = dispatcher(args.utterance)
            print(f"  {call}   confidence {call.confidence:.2f}   {call.elapsed_ms:.0f}ms")
            sys.exit(0)
        sys.exit(handle(dispatcher, args.utterance, args.threshold))

    print("jev — type a command (Ctrl-D to quit)")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"quit", "exit", ":q"}:
            break
        if args.dry_run:
            call = dispatcher(line)
            print(f"  {call}   confidence {call.confidence:.2f}   {call.elapsed_ms:.0f}ms")
        else:
            handle(dispatcher, line, args.threshold)


if __name__ == "__main__":
    main()
