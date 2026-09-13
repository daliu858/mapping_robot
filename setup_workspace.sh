#!/usr/bin/env bash
# Assemble a complete catkin workspace src/ from the pinned upstream repos,
# the local patches, and the overlay files in this repository.
#
# Usage: ./setup_workspace.sh [TARGET_SRC_DIR]   (default: ./catkin_ws/src)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$HERE/catkin_ws/src}"

JETBOT_PRO_COMMIT=b76d483fbd5ceb41a66be4945ed2fc712b5d36a4
GSCAM_COMMIT=1b0b8e51b91522edadd8399d4ad920f8afbc487d

mkdir -p "$TARGET"

clone_pin () { # url dir commit
  if [ ! -d "$2/.git" ]; then
    git clone "$1" "$2"
  fi
  git -C "$2" fetch --quiet
  git -C "$2" checkout --quiet "$3"
}

echo "==> cloning upstream packages (pinned)"
clone_pin https://github.com/waveshare/jetbot_pro   "$TARGET/jetbot_pro" "$JETBOT_PRO_COMMIT"
clone_pin https://github.com/ros-drivers/gscam      "$TARGET/gscam"      "$GSCAM_COMMIT"

echo "==> applying local patches"
git -C "$TARGET/jetbot_pro" apply --whitespace=nowarn "$HERE/patches/jetbot_pro_local_changes.patch"
git -C "$TARGET/gscam"      apply --whitespace=nowarn "$HERE/patches/gscam_local_changes.patch"

echo "==> copying overlay files"
cp -r "$HERE/overlay/jetbot_pro/." "$TARGET/jetbot_pro/"
cp -r "$HERE/overlay/gscam/."      "$TARGET/gscam/"

echo "==> done. Build with:"
echo "    cd $(dirname "$TARGET") && catkin_make"
