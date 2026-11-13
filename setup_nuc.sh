#!/usr/bin/env bash
# Set up this repo on the hackathon NUC (Linux, Intel Core Ultra: CPU + Arc GPU + NPU).
#
#   ./setup_nuc.sh            # runtime only (OpenVINO + GenAI + Viam): ~1 min, no torch
#   ./setup_nuc.sh --train    # + CPU-only torch/anomalib so you can also train on the NUC
#
# Then copy models/<name>/ over (scp -r models/<name> nuc:~/sorter/models/) and run
#   python benchmark.py --models models/*/model.xml
set -euo pipefail
cd "$(dirname "$0")"

PY=$(command -v python3.11 || command -v python3.12 || command -v python3)
echo "using $PY ($($PY --version))"
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip

if [[ "${1:-}" == "--train" ]]; then
  # CPU wheels: the default Linux torch pulls ~3 GB of CUDA libraries we can't use.
  pip install -q torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
  pip install -q -r requirements.txt
else
  pip install -q -r requirements-runtime.txt
fi

echo
python - <<'EOF'
import openvino as ov
core = ov.Core()
print("OpenVINO", ov.__version__)
for d in core.available_devices:
    print(f"  {d:8} {core.get_property(d, 'FULL_DEVICE_NAME')}")
missing = {"GPU", "NPU"} - {d.split(".")[0] for d in core.available_devices}
if missing:
    print("missing:", missing, "-> GPU needs intel-opencl-icd/level-zero, NPU needs the intel NPU driver;"
          " ask a mentor, the provided pipeline image should have them")
EOF
echo
[ -f api-key.json ] || echo 'next: create api-key.json  {"address": "...", "api_key": "...", "api_key_id": "..."}'
echo "next: python sorter.py resources"
