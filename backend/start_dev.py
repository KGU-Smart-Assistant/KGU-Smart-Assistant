"""
개발용 서버: app/ 디렉터리만 reload 감시 (.venv·site-packages 제외).

  python start_dev.py

CLI에서 `uvicorn app.main:app --reload`만 쓰면 프로젝트 루트 전체가 감시되어
`.venv` 변경으로 무한 reload가 날 수 있습니다.
"""
from __future__ import annotations

from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
APP_DIR = str((ROOT / "app").resolve())

if __name__ == "__main__":
    # reload_dirs만 app으로 제한하면 .venv는 감시 대상이 아님.
    # reload_excludes와 함께 쓰면 일부 uvicorn/Windows 조합에서
    # resolve_reload_patterns()가 오래 걸리거나 예외가 날 수 있어 생략함.
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[APP_DIR],
    )
