#!/usr/bin/env bash
# SPINE verification protocol.
#
# Runs, in order:
#   1. the full pytest suite
#   2. identity, dpmlm and spine modes end-to-end on the synthetic data
#   3. the offline harness on each output
#   4. two consecutive PRODUCTION spine runs, asserting Text differs while
#      ID/Author/HS stay identical
#
# Uses the deterministic core path (mlm.backend=hash via configs/test.yaml,
# proxy classifiers) so it passes on a fresh CORE install with NO downloads:
#     pip install -r requirements.txt && pip install -e .
# For the real open-weight models, see README (requirements-hf.txt); this script
# deliberately does not depend on them.
#
# Override the interpreter with:  PYTHON=path/to/python scripts/verify.sh
set -u

PYTHON="${PYTHON:-python}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
CONFIG=configs/test.yaml
DATA=data/synthetic_dev.csv
OUT=verify_out
mkdir -p "$OUT"

fail() { echo ""; echo "VERIFY FAILED: $1"; exit 1; }
hr()   { echo "----------------------------------------------------------------------"; }

hr; echo "[0/4] regenerate synthetic data"; hr
$PYTHON scripts/make_synthetic.py || fail "synthetic data generation"

hr; echo "[1/4] full pytest suite"; hr
$PYTHON -m pytest -q || fail "pytest suite"

hr; echo "[2/4] identity / dpmlm / spine end-to-end"; hr
for mode in identity dpmlm spine; do
  $PYTHON -m wrapper.run --in "$DATA" --out "$OUT/$mode.csv" --mode "$mode" \
    --config "$CONFIG" --debug-seed VERIFY --logs "$OUT/$mode.log.jsonl" \
    || fail "$mode mode run (incl. built-in diff check)"
done

hr; echo "[3/4] harness on each output (LOCAL trade-off approximation)"; hr
for mode in identity dpmlm spine; do
  echo ">>> harness on $mode"
  $PYTHON -m harness.evaluate --original "$DATA" --privatized "$OUT/$mode.csv" \
    --config "$CONFIG" --json "$OUT/$mode.report.json" || fail "harness on $mode"
done

hr; echo "[4/4] two consecutive PRODUCTION spine runs (no debug seed)"; hr
$PYTHON -m wrapper.run --in "$DATA" --out "$OUT/spine_run1.csv" --mode spine \
  --config "$CONFIG" --logs "$OUT/run1.log.jsonl" || fail "spine production run 1"
$PYTHON -m wrapper.run --in "$DATA" --out "$OUT/spine_run2.csv" --mode spine \
  --config "$CONFIG" --logs "$OUT/run2.log.jsonl" || fail "spine production run 2"

$PYTHON - "$DATA" "$OUT/spine_run1.csv" "$OUT/spine_run2.csv" <<'PY' || fail "production independence check"
import sys
from wrapper.csvio import read_csv
_, o = read_csv(sys.argv[1])
_, a = read_csv(sys.argv[2])
_, b = read_csv(sys.argv[3])
assert len(o) == len(a) == len(b), "row counts differ"
text_diffs = sum(1 for x, y in zip(a, b) if x["Text"] != y["Text"])
for run_name, rows in (("run1", a), ("run2", b)):
    for orig, got in zip(o, rows):
        for col in ("ID", "Author", "HS"):
            assert orig[col] == got[col], f"{run_name}: {col} changed"
print(f"  ID/Author/HS identical across both runs and input.")
print(f"  Text differs in {text_diffs}/{len(o)} rows between the two runs.")
assert text_diffs > 0, "two production runs produced identical Text everywhere"
PY

hr
echo "VERIFY PASSED — all steps completed. Artifacts in $OUT/"
hr
