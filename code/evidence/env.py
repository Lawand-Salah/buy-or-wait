"""Minimal .env loading -- no python-dotenv dependency.

AGENTS.md 6.4: "Read secrets from environment variables only." A .env file
next to it (also AGENTS.md 4: "Use environment variables and a .env file
when needed") is the standard way to get a secret into the environment
without exporting it in every terminal; this just parses that file into
``os.environ`` so the rest of the evidence package can read
``ANTHROPIC_API_KEY`` the normal way, regardless of whether the shell that
launched this process happened to have it exported.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: "str | Path | None" = None) -> None:
    """Parses ``KEY=value`` lines from a .env file into ``os.environ``.

    Never overrides a variable the environment already has -- an
    explicitly exported value always wins over the file. Missing file,
    blank lines, and ``#`` comments are silently fine; malformed lines
    (no ``=``) are skipped rather than raised on, since a secrets file
    failing to parse should never crash the program that needs the secret.
    """
    if path is None:
        # repo_root/.env -- this file lives at repo_root/code/evidence/env.py
        path = Path(__file__).resolve().parents[2] / ".env"
    path = Path(path)
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
