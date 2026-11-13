"""Run every offline test (~1 min). Use after any edit tomorrow:  python tests/run_all.py"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    ["decider.py"],
    ["settings.py"],
    ["tests/fake_machine.py"],
    ["tests/test_sorter_fake.py"],
    ["tests/test_dashboard.py"],
    ["tests/test_doctor.py"],
    ["sorter.py", "offline", "--limit", "12", "--no-artifacts"],
]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        t0 = time.time()
        r = subprocess.run([sys.executable, *t], cwd=ROOT, capture_output=True, text=True)
        ok = r.returncode == 0
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {' '.join(t):48} {time.time() - t0:5.1f}s")
        if not ok:
            print("\n".join((r.stdout + r.stderr).strip().splitlines()[-15:]))
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
