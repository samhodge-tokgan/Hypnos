#!/usr/bin/env bash
# Download the MEMatte ONNX models into the plugin bundle's Contents/Resources.
#
# Release installers are shipped MODEL-LESS to stay under GitHub's per-asset size
# limit, exactly as the humbaba project does. Run this once after installing:
#
#   bash Contents/fetch_models.sh            # from inside the .ofx.bundle
#   bash tools/fetch_models.sh --dest DIR    # or anywhere, for a dev build
#
# You can skip this entirely by exporting the models yourself
# (tools/export_mematte.py) and pointing $MEMATTE_MODEL_DIR at them, or by
# setting the plugin's Backbone/Decoder file parameters.
set -euo pipefail

BASE_URL="${MEMATTE_MODEL_BASE_URL:-https://github.com/samhodge-tokgan/Hypnos/releases/download/models-v1}"
VARIANTS="s b"
DEST=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2;;
    --variant) VARIANTS="$2"; shift 2;;
    --base-url) BASE_URL="$2"; shift 2;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

if [ -z "$DEST" ]; then
  # Default to Resources/ next to this script (the in-bundle case).
  DEST="$(cd "$(dirname "$0")" && pwd)/Resources"
fi
mkdir -p "$DEST"

# SHA-256 of each published asset. Filled in when the models-v1 release is cut;
# until then the checksum step is skipped with a loud warning rather than
# silently trusting whatever the URL returns.
sha_for() {
  case "$1" in
    mematte_s_backbone.onnx) echo "";;
    mematte_s_decoder.onnx)  echo "";;
    mematte_b_backbone.onnx) echo "";;
    mematte_b_decoder.onnx)  echo "";;
    *) echo "";;
  esac
}

verify() {
  local file="$1" want="$2"
  if [ -z "$want" ]; then
    echo "  WARNING: no pinned checksum for $(basename "$file") - NOT verified" >&2
    return 0
  fi
  local got
  if command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "$file" | cut -d' ' -f1)"
  else
    got="$(shasum -a 256 "$file" | cut -d' ' -f1)"
  fi
  if [ "$got" != "$want" ]; then
    echo "  CHECKSUM MISMATCH for $(basename "$file")" >&2
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
    echo "fetching $name"
    curl -fL --progress-bar -o "$out.part" "$BASE_URL/$name"
    verify "$out.part" "$(sha_for "$name")"
    mv "$out.part" "$out"
  done
done

echo
echo "models are in: $DEST"
ls -la "$DEST"
