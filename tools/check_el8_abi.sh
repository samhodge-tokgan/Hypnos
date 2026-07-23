#!/usr/bin/env bash
# Fail if a Linux binary needs glibc/libstdc++ symbols newer than the VFX
# Reference Platform baseline (RHEL/Rocky 8: glibc 2.28, GLIBCXX_3.4.25).
#
# Why this exists: v0.1.0 shipped a Linux .ofx built on Ubuntu 24.04. It needed
# GLIBC_2.33/2.34/2.38 and GLIBCXX_3.4.26, so it would not load AT ALL on Rocky
# 8 - the platform most facilities actually run. Nothing in the build or the
# test suite noticed; it was caught only by trying it on a real Rocky 8 box.
#
# Build inside manylinux_2_28 (or on EL8 with gcc-toolset-12) and run this to
# keep that from happening again. A binary that passes here also runs on newer
# distros - the compatibility only goes one way.
#
# Usage:  tools/check_el8_abi.sh build/MEMatte.ofx.bundle/Contents/Linux-x86-64/MEMatte.ofx
set -euo pipefail

MAX_GLIBC="${MAX_GLIBC:-2.28}"
MAX_GLIBCXX="${MAX_GLIBCXX:-3.4.25}"

[ $# -ge 1 ] || { echo "usage: $0 <binary> [binary...]" >&2; exit 2; }

# Newest version of a given symbol family required by the binary.
newest() { grep "^$1_" "$2" | sed "s/^$1_//" | sort -V | tail -1; }

# a <= b, comparing dotted versions.
le() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$2" ]; }

status=0
for bin in "$@"; do
  if [ ! -f "$bin" ]; then
    echo "$bin: not found" >&2
    status=1
    continue
  fi
  vers=$(mktemp)
  objdump -T "$bin" 2>/dev/null | grep -oE 'GLIBC(XX)?_[0-9.]+' | sort -Vu > "$vers"

  gl=$(newest GLIBC "$vers" || true)
  gx=$(newest GLIBCXX "$vers" || true)
  bad=0

  if [ -n "$gl" ] && ! le "$gl" "$MAX_GLIBC"; then
    echo "$bin: requires GLIBC_$gl, newer than the EL8 floor GLIBC_$MAX_GLIBC" >&2
    bad=1
  fi
  if [ -n "$gx" ] && ! le "$gx" "$MAX_GLIBCXX"; then
    echo "$bin: requires GLIBCXX_$gx, newer than the EL8 floor GLIBCXX_$MAX_GLIBCXX" >&2
    bad=1
  fi

  if [ "$bad" = 0 ]; then
    echo "$bin: OK (max GLIBC_${gl:-none}, GLIBCXX_${gx:-none})"
  else
    echo "$bin: would NOT load on Rocky 8 - build in manylinux_2_28 / gcc-toolset-12" >&2
    status=1
  fi
  rm -f "$vers"
done

exit $status
