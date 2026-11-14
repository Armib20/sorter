#!/usr/bin/env bash
# Fetch the large files git doesn't track (they're in .gitignore).
#
#   ./fetch_models.sh            # smoke-test anomaly model (107 MB, GitHub release, checksum verified)
#   ./fetch_models.sh --vlm      # + InternVL2-1B int4 (0.8 GB, Hugging Face)
#   ./fetch_models.sh --vlm-4b   # + Qwen3-VL-4B int4 (3.1 GB, Hugging Face)
#
# Run setup_nuc.sh first: the VLM downloads use `hf` from .venv. Skips anything already present.
# Datasets are not fetched here: see README "Large files".
set -euo pipefail
cd "$(dirname "$0")"

REPO=Armib20/sorter
SMOKE=patchcore_bottle_c001
SMOKE_SHA256=8aece3e1cc1b529f108ebb666256d2bbb53ee6c218d698385450e0b75da65441
HF=.venv/bin/hf
[ -x "$HF" ] || HF=hf

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

if [ -f "models/$SMOKE/model.xml" ]; then
  echo "ok   models/$SMOKE (already present)"
else
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  url="https://github.com/$REPO/releases/download/model-$SMOKE/$SMOKE.tar.gz"
  echo "get  $url"
  curl -fL --progress-bar -o "$tmp/m.tgz" "$url"
  [ "$(sha256 "$tmp/m.tgz")" = "$SMOKE_SHA256" ] || { echo "checksum mismatch for $SMOKE" >&2; exit 1; }
  mkdir -p models && tar -xzf "$tmp/m.tgz" -C models
  echo "ok   models/$SMOKE"
fi

hf_get() {  # repo_id local_dir [extra hf args...]
  local repo=$1 dir=$2; shift 2
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then echo "ok   $dir (already present)"; return; fi
  command -v "$HF" >/dev/null || { echo "need the hf CLI: run ./setup_nuc.sh first" >&2; exit 1; }
  "$HF" download "$repo" --local-dir "$dir" "$@"
  echo "ok   $dir"
}

for arg in "$@"; do
  case "$arg" in
    --vlm)    hf_get OpenVINO/InternVL2-1B-int4-ov models/vlm-internvl2-1b ;;
    --vlm-4b) hf_get OpenVINO/Qwen3-VL-4B-Instruct-int4-ov models/vlm-qwen3-4b ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done
