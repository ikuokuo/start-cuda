#!/bin/bash

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

find "$BASE_DIR" -iname "_build" -type d -print0 | while IFS= read -r -d "" subdir; do
  if [[ $subdir == $BASE_DIR* ]]; then
    echo "Remove $subdir"
    rm -rf "$subdir"
  fi
done
