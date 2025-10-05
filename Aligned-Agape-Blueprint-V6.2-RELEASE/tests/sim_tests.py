
import argparse, subprocess, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=3)
    args = ap.parse_args()
    cmd = [sys.executable, str(ROOT / "main.py"), "--turns", str(args.turns)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    out = sorted((ROOT/"out").glob("turn_*.json"))
    assert out, "No turn outputs"
    data = json.loads(out[-1].read_text(encoding="utf-8"))
    assert "J" in data and "SCM" in data, "Missing metrics"
    print("Smoke test OK. Last file:", out[-1].name)

if __name__ == "__main__":
    main()
