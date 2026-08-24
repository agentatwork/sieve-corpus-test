#!/bin/bash
cd /home/agent/work/sieve-test
for i in $(seq 1 300); do
  [ -f scores_sg_hard.json ] && break
  sleep 10
done
python3 score.py --set graphics
