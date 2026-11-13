#!/usr/bin/env bash
# usage: run3.sh MODEL EPOCHS SPLIT TAG FOLDS...   runs one model, all given folds concurrently
S=$(dirname "$(readlink -f "$0")")
PY=/home/charlles/github_repos/chemprop-contrib/.claude/worktrees/reaction-parity/.venv/bin/python
MODEL=$1; EPOCHS=$2; SPLIT=$3; TAG=$4; shift 4
mkdir -p "$S/$TAG/logs"
for fold in "$@"; do
  "$PY" "$S/bench2.py" "$MODEL" "$fold" "$SPLIT" "$EPOCHS" "$S/$TAG/$MODEL-f$fold" > "$S/$TAG/logs/$MODEL-f$fold.log" 2>&1 &
done
wait
