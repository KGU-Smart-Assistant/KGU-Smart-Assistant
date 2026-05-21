from __future__ import annotations

import os
import shlex
import sys
import time
from typing import Sequence

from app.crawlers import run_ingest
from app.crawlers.job_lock import DEFAULT_CRAWLER_LOCK_NAME, postgres_advisory_lock


def default_ingest_args() -> list[str]:
    raw_args = os.getenv(
        "CRAWLER_INGEST_ARGS",
        "--store-db --store-vectors",
    )
    return shlex.split(raw_args)


def run_once(*, lock_name: str, ingest_args: Sequence[str]) -> bool:
    with postgres_advisory_lock(lock_name) as acquired:
        if not acquired:
            print(
                f"crawler_lock=skipped lock_name={lock_name} reason=already_running",
                flush=True,
            )
            return False
        print(f"crawler_lock=acquired lock_name={lock_name}", flush=True)
        run_ingest.main(list(ingest_args))
        print(f"crawler_lock=released lock_name={lock_name}", flush=True)
        return True


def main(argv: Sequence[str] | None = None) -> None:
    lock_name = os.getenv("CRAWLER_LOCK_NAME", DEFAULT_CRAWLER_LOCK_NAME)
    ingest_args = list(argv or []) or default_ingest_args()
    run_once(lock_name=lock_name, ingest_args=ingest_args)
    if os.getenv("CRAWLER_KEEP_ALIVE", "false").casefold() in {"1", "true", "yes"}:
        print("crawler_keep_alive=true", flush=True)
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main(sys.argv[1:])
