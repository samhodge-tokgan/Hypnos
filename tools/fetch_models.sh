#!/usr/bin/env bash
# Put the MEMatte ONNX models into the plugin bundle's Contents/Resources.
#
# THIS PROJECT DOES NOT REDISTRIBUTE MODEL WEIGHTS, and there is no default
# download URL. That is deliberate, not an oversight:
#
#   * MEMatte licenses its CODE as MIT ("The code is released under the MIT
#     License"); it says nothing about the weights and ships no LICENSE file.
#   * The published checkpoints (MEMatte_ViT{S,B}_DIM.pth) are trained on the
#     Adobe Deep Image Matting / Composition-1k dataset, and fine-tuned from
#     ViTMatte's Composition-1k checkpoints. Adobe's dataset agreement restricts
#     models trained on it to NON-COMMERCIAL use and distribution.
#
# So exporting the models is something you do yourself, having accepted the
# upstream terms. It takes about a minute per variant:
#
#     git clone https://github.com/linyiheng123/MEMatte
#     pip install -r tools/requirements.txt
#     pip install 'git+https://github.com/facebookresearch/detectron2.git'
#     python3 tools/export_mematte.py --mematte-repo ../MEMatte \
#         --checkpoint ../MEMatte/checkpoints/MEMatte_ViTS_DIM.pth \
#         --variant s --out-dir build/models
#
# Then either copy the .onnx files into this bundle's Contents/Resources, point
# $MEMATTE_MODEL_DIR at them, or set the plugin's Backbone/Decoder file params.
#
# If your facility mirrors the exported models internally, point this script at
# that mirror and it will fetch and (optionally) checksum them:
#
#     bash fetch_models.sh --base-url https://internal.example/mematte
#     MEMATTE_MODEL_BASE_URL=https://internal.example/mematte bash fetch_models.sh
set -euo pipefail

BASE_URL="${MEMATTE_MODEL_BASE_URL:-}"
VARIANTS="s b"
DEST=""
SHA_FILE=""

# Print the header comment (everything from line 2 up to the first non-comment).
usage() { awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2;;
    --variant) VARIANTS="$2"; shift 2;;
    --base-url) BASE_URL="$2"; shift 2;;
    --sha256-file) SHA_FILE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

if [ -z "$BASE_URL" ]; then
  usage
  echo
  echo "No --base-url given, and this project publishes no model weights." >&2
  echo "Export them yourself with tools/export_mematte.py (see above)." >&2
  exit 1
fi

if [ -z "$DEST" ]; then
  # Default to Resources/ next to this script (the in-bundle case).
  DEST="$(cd "$(dirname "$0")" && pwd)/Resources"
fi
mkdir -p "$DEST"

verify() {
  local file="$1" name="$2"
  if [ -z "$SHA_FILE" ]; then
    echo "  note: no --sha256-file given, checksum NOT verified" >&2
    return 0
  fi
  local want
  want="$(awk -v n="$name" '$2 == n || $2 == "*"n {print $1}' "$SHA_FILE" | head -1)"
  if [ -z "$want" ]; then
    echo "  note: $name not listed in $SHA_FILE, checksum NOT verified" >&2
    return 0
  fi
  local got
  if command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "$file" | cut -d' ' -f1)"
  else
    got="$(shasum -a 256 "$file" | cut -d' ' -f1)"
  fi
  if [ "$got" != "$want" ]; then
    echo "  CHECKSUM MISMATCH for $name" >&2
    echo "    expected $want" >&2
    echo "    got      $got" >&2
    rm -f "$file"
    return 1
  fi
  echo "  checksum ok"
}

for v in $VARIANTS; do
  for part in backbone decoder; do
    name="mematte_${v}_${part}.onnx"
    out="$DEST/$name"
    if [ -f "$out" ]; then
      echo "$name already present, skipping"
      continue
    fi
    echo "fetching $name from $BASE_URL"
    curl -fL --progress-bar -o "$out.part" "$BASE_URL/$name"
    verify "$out.part" "$name"
    mv "$out.part" "$out"
  done
done

echo
echo "models are in: $DEST"
ls -la "$DEST"
