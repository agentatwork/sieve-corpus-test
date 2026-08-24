#!/bin/bash
# Wait on the artefact, never on a process name: a pgrep pattern that matches
# this script's own command line waits forever.
cd /home/agent/work/sieve-test
for i in $(seq 1 240); do
  [ -f scores_hard.json ] && break
  sleep 10
done
for s in sg_clean sg_web sg_hard; do python3 score.py --set "$s"; done
