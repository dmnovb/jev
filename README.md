# jev

Local macOS assistant. Jev picks a typed action from a closed catalog; your machine runs it.

```bash
uv sync
uv run jev "open notes"
uv run jev   # REPL
```

Requires `TYPESAFE_API_KEY` in `.env` (see `.env.example`).

Low-confidence calls are refused instead of executed. Tune with `--threshold`.
