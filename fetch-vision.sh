#!/bin/sh
# Download the vision model baked into the GPU helper (docker/gpu-helper/Dockerfile):
# Gemma 4 31B, 4-bit (about 18 GB), and its image encoder (about 1 GB).
#
# Files land in docker/models/vision/ (git-ignored). Building the image needs them
# there; running it needs no network at all. Downloaded files are kept, and checked
# against the checksums below.
set -eu
cd "$(dirname "$0")"
mkdir -p models/vision
cd models/vision
REPO=https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

fetch() { # file sha256
  if [ -s "$1" ]; then
    echo "have $1"
  else
    echo "fetching $1"
    curl -fL --retry 3 -C - -o "$1.part" "$REPO/$1"
    mv "$1.part" "$1"
  fi
  got="$(sha256 "$1")"
  if [ "$got" != "$2" ]; then
    echo "checksum mismatch for $1: $got (expected $2)" >&2
    exit 1
  fi
  echo "checksum ok: $1"
}

fetch gemma-4-31B-it-Q4_K_M.gguf 38bd64c852c4b460434cc7162fa9bdcf242faf86502581a754cb72956bb17f84
fetch mmproj-F16.gguf 6edcca228213c28d3567a35d22f849eea52d8360875093851959adf5d2f270eb
ls -lh ./*.gguf
