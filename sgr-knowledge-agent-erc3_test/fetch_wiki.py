"""
Utility script to dump all wiki documents into the local docs/ folder.

Usage:
    python fetch_wiki.py

Requires ERC3_API_KEY in the environment (loaded from .env if present).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from erc3 import ERC3, ApiException


def main() -> None:
    load_dotenv()

    out_dir = Path(__file__).parent / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start a short-lived session to access the wiki APIs.
    core = ERC3()
    session = core.start_session(
        benchmark="erc3-dev",
        workspace=os.getenv("ERC3_WORKSPACE", "local"),
        name="wiki-downloader",
        architecture="utility-script",
    )

    status = core.session_status(session.session_id)
    if not status.tasks:
        raise RuntimeError("No tasks returned in session, cannot access wiki.")

    task = status.tasks[0]
    core.start_task(task)
    api = core.get_erc_client(task)

    wiki = api.list_wiki()
    print(f"Found {len(wiki.paths)} wiki files (sha1={wiki.sha1}).")

    for path in wiki.paths:
        try:
            loaded = api.load_wiki(path)
        except ApiException as exc:
            print(f"Failed to load {path}: {exc.detail}")
            continue

        target = out_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(loaded.content, encoding="utf-8")
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
