"""Run every test file in its own process.

TensorFlow and PyTorch can crash when loaded into the same Python process, so
the Keras and PyTorch tests must not share one pytest run.

    python scripts/run_tests.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    failed = []
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        print(f"\n=== {f.name} ===", flush=True)
        if subprocess.run([sys.executable, "-m", "pytest", "-q", str(f)], cwd=ROOT).returncode:
            failed.append(f.name)
    print("\nAll test files passed." if not failed else f"\nFailed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
