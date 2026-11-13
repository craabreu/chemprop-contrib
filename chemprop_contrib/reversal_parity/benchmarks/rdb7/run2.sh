#!/usr/bin/env bash
# usage: run2.sh EPOCHS SPLIT TAG FOLDS...   runs the four models concurrently, one fold at a time
S=$(dirname "$(readlink -f "$0")")
PY=/home/charlles/github_repos/chemprop-contrib/.claude/worktrees/reaction-parity/.venv/bin/python
EPOCHS=$1; SPLIT=$2; TAG=$3; shift 3
mkdir -p "$S/$TAG/logs"
for fold in "$@"; do
  for m in stock stock_aug parity parity_marcus; do
    "$PY" "$S/bench2.py" "$m" "$fold" "$SPLIT" "$EPOCHS" "$S/$TAG/$m-f$fold" > "$S/$TAG/logs/$m-f$fold.log" 2>&1 &
  done
  wait
done
