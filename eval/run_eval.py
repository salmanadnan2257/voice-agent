"""Standalone runner: `python -m eval.run_eval`

Runs every scripted scenario in eval/scenarios.py against the real
conversation state machine and a real (temporary) SQLite calendar, using the
free deterministic ScriptedIntentEngine instead of the real Gemini API, and
prints a pass/fail report.
"""
from __future__ import annotations

import sys

from .scenarios import SCENARIOS, run_all


def main() -> int:
    results = run_all()
    passed = 0
    for name, category, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] ({category}) {name}")
        if not ok:
            print(f"         {detail}")
        passed += int(ok)

    total = len(results)
    print(f"\n{passed}/{total} scenarios passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
