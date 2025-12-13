"""
Utility script to dump all wiki documents into the local docs/ folder
and build an index (paths + hashes) for downstream consumers like the
security checker.

Usage:
    python fetch_wiki.py

Requires ERC3_API_KEY in the environment (loaded from .env if present).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from erc3 import ERC3, ApiException


def build_index_entry(path: str, content: str) -> Dict[str, Any]:
    payload = content.encode("utf-8")
    return {
        "path": path,
        "sha1": hashlib.sha1(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def write_index(out_dir: Path, entries: List[Dict[str, Any]], tree_sha1: str | None) -> Path:
    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tree_sha1": tree_sha1,
        "files": entries,
    }
    target = out_dir / "wiki_index.json"
    target.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


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
    print(f"Found {len(wiki.paths)} wiki files (sha1={getattr(wiki, 'sha1', None)}).")

    index_entries: List[Dict[str, Any]] = []

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

        index_entries.append(build_index_entry(path, loaded.content))

    index_path = write_index(out_dir, index_entries, getattr(wiki, "sha1", None))
    print(f"Wrote index {index_path}")


if __name__ == "__main__":
    main()
