"""Typed macOS actions with closed-set arguments."""

from __future__ import annotations

import ast
import operator
import re
import subprocess
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

AppName = Literal[
    "Notes",
    "Safari",
    "Terminal",
    "Finder",
    "Messages",
    "Mail",
    "Calendar",
    "Music",
    "Photos",
    "Reminders",
    "System Settings",
    "Cursor",
    "Ghostty",
    "Obsidian",
    "Notion",
    "Slack",
    "Spotify",
    "Chrome",
    "Comet",
    "ChatGPT",
]

UrlPreset = Literal[
    "github",
    "gmail",
    "calendar",
    "youtube",
    "twitter",
    "docs",
]

FolderName = Literal[
    "Desktop",
    "Downloads",
    "Documents",
    "Home",
]

BrowserName = Literal["Comet", "Safari", "Chrome"]

APP_BUNDLE: dict[str, str] = {
    "Notes": "Notes",
    "Safari": "Safari",
    "Terminal": "Terminal",
    "Finder": "Finder",
    "Messages": "Messages",
    "Mail": "Mail",
    "Calendar": "Calendar",
    "Music": "Music",
    "Photos": "Photos",
    "Reminders": "Reminders",
    "System Settings": "System Settings",
    "Cursor": "Cursor",
    "Ghostty": "Ghostty",
    "Obsidian": "Obsidian",
    "Notion": "Notion",
    "Slack": "Slack",
    "Spotify": "Spotify",
    "Chrome": "Google Chrome",
    "Comet": "Comet",
    "ChatGPT": "ChatGPT",
}

URLS: dict[str, str] = {
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "calendar": "https://calendar.google.com",
    "youtube": "https://www.youtube.com",
    "twitter": "https://x.com",
    "docs": "https://docs.typesafe.ai",
}

FOLDERS: dict[str, Path] = {
    "Desktop": Path.home() / "Desktop",
    "Downloads": Path.home() / "Downloads",
    "Documents": Path.home() / "Documents",
    "Home": Path.home(),
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _run(cmd: list[str]) -> str:
    subprocess.run(cmd, check=True)
    return " ".join(cmd)


def extract_search_query(utterance: str) -> str:
    """Pull a search string out of a natural-language request. Free text — not Jev."""
    q = utterance.strip()
    q = re.sub(
        r"(?i)^(can you |could you |would you |please )?(search( for)?|google|look up|find)\s+",
        "",
        q,
    )
    q = re.sub(
        r"(?i)\s+(on|in|using|with|via)\s+(comet|safari|chrome|google chrome)\??\s*$",
        "",
        q,
    )
    q = q.strip(" ?")
    return q or utterance.strip()


def extract_math_expression(utterance: str) -> str:
    """Pull an arithmetic expression from a natural-language request. Free text — not Jev."""
    text = utterance.strip()
    text = re.sub(
        r"(?i)^(what('?s| is| are)|whats|calculate|compute|solve|eval(uate)?)\s+",
        "",
        text,
    )
    text = text.strip(" ?=")
    match = re.search(r"[\d.]+(?:\s*[+\-*/^%×÷]\s*[\d.]+)+", text)
    expr = match.group(0) if match else text
    return (
        expr.replace("×", "*")
        .replace("÷", "/")
        .replace("^", "**")
        .replace(" ", "")
    )


def _eval_ast(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_ast(node.operand))
    raise ValueError("unsupported expression")


def open_app(app: AppName) -> str:
    """Launch or focus a macOS application."""
    bundle = APP_BUNDLE[app]
    return _run(["open", "-a", bundle])


def open_url(preset: UrlPreset) -> str:
    """Open a known website in the default browser."""
    return _run(["open", URLS[preset]])


def open_folder(folder: FolderName) -> str:
    """Reveal a common folder in Finder."""
    return _run(["open", str(FOLDERS[folder])])


def web_search(browser: BrowserName = "Comet", query: str = "") -> str:
    """Search the web in a browser. `query` is free text from the utterance (not a Jev choice)."""
    if not query:
        raise ValueError("web_search requires a query")
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    bundle = APP_BUNDLE[browser]
    return _run(["open", "-a", bundle, url])


def calculate(expression: str = "") -> str:
    """Evaluate a basic arithmetic expression locally and return the result."""
    if not expression:
        raise ValueError("calculate requires an expression")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_ast(tree)
    except Exception as exc:
        return f"could not calculate {expression!r}: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"


def clarify() -> str:
    """Ask the user to rephrase — used when the request is not an executable action."""
    return (
        "Please rephrase — I only run known local actions "
        "(open app, URL, folder, web search, or calculate)."
    )


TOOLS: dict[str, object] = {
    "open_app": open_app,
    "open_url": open_url,
    "open_folder": open_folder,
    "web_search": web_search,
    "calculate": calculate,
    "clarify": clarify,
}
