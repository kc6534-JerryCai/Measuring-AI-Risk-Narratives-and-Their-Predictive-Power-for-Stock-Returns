"""Safe project-root launcher for the standalone X AI news downloader."""

from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "moduals" / "x_ai_news_download.py"
GLOBALS = {"__name__": "__main__", "__file__": str(SCRIPT)}
exec(compile(SCRIPT.read_bytes(), str(SCRIPT), "exec"), GLOBALS)
