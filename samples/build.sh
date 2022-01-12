#!/bin/bash

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

function build() {
  local subdir="$1"; shift
  local options="$1"; shift
  if [[ -z "$options" ]]; then
    options="BIN_DIR=../_build/bin OBJ_DIR=../_build/obj"
  fi
  echo -e "\033[1;35m$ cd $(basename "$subdir")/\033[0m"
  (cd "$subdir" || exit; make $options)
}

# build

subdirs=(
  quick_start
  matmul
  #img_proc
  npp
  nvjpeg
  tensorrt
)

for subdir in "${subdirs[@]}"; do
  build "$BASE_DIR/$subdir"
done

# build Chapter*

# Bash Pitfalls: http://mywiki.wooledge.org/BashPitfalls
# http://askubuntu.com/questions/343727/filenames-with-spaces-breaking-for-loop-find-command
# find "$BASE_DIR" -name "Chapter*" -maxdepth 1 -type d -print0 | while IFS= read -r -d "" subdir; do
#   build "$subdir"
# done
