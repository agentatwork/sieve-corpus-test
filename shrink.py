#!/usr/bin/env python3
"""Why does more TTA raise AUC everywhere but lower accuracy at the shipped threshold?

The pre-registered result is that variant A fails P4, and AUC says why: A barely changes
discrimination at all (|dAUC| <= 0.002). But variant B -- always average both views --
raises AUC on all three conditions (+0.009 / +0.005 / +0.006) while *losing* balanced
accuracy at 0.65 on two of them. A change cannot both rank better and classify worse
unless the threshold has moved relative to the score distribution.

Hypothesis: averaging two logits is a variance-reduction operation. mean(z1, z2) has
smaller spread than z1 whenever the two views disagree at all, so the whole distribution
contracts toward its centre while the calibration constant (bias +0.43, temp 1.0) stays
where it was fitted for the single-view distribution. Recall dies first, because AI images
sit in the upper tail and the upper tail is what contraction pulls in.

Two tests, both on data already on disk:

  1. Direct: does the logit spread actually contract, and does the AI tail move down more
     than the real tail moves up?
  2. Held out: re-fit ONE number -- the decision threshold -- on ONE condition, then apply
     it unchanged to the other two. If the diagnosis is right, a threshold fitted on clean
     should recover the loss on web and hard without being fitted to either. If it only
     works on the condition it was fitted on, the diagnosis is wrong and this is overfitting.

Test 2 is the honest version of "B's AUC is higher so some threshold must be better."
That statement is true by construction and worth nothing on its own; transfer is what
makes it a claim about Sieve rather than about my corpus.
"""
import os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variants as V

CONDS = ("clean", "web", "hard")


def logits(recs, variant):
    """The pre-calibration logit each variant feeds to the sigmoid."""
    out = []
    for r in recs:
        s = V.apply(r, variant)
        s = min(max(s, 1e-12), 1 - 1e-12)
        out.append(V.math.log(s / (1 - s)) - V.BIAS)   # undo cal() to recover z
    return out


def ba_at(recs, variant, thr):
    ai = [V.apply(r, variant) for r in recs if r["label"] == 1]
    real = [V.apply(r, variant) for r in recs if r["label"] == 0]
    rec = 100.0 * sum(s >= thr for s in ai) / len(ai)
    spec = 100.0 * sum(s < thr for s in real) / len(real)
    return (rec + spec) / 2


def best_thr(recs, variant):
    """Threshold maximising BA on this condition. Only ever fitted on the *source* set."""
    cands = sorted({round(V.apply(r, variant), 4) for r in recs})
    return max(cands, key=lambda t: ba_at(recs, variant, t))


def main():
    print("TEST 1 -- does the score distribution contract?\n")
    print(f"  {'cond':6} {'variant':9} {'sd(z) all':>10} {'mean z AI':>10} {'mean z real':>12}")
    for cond in CONDS:
        d = V.load(cond)
        if d is None:
            continue
        for v in ("baseline", "B"):
            zs = logits(d, v)
            zai = [z for z, r in zip(zs, d) if r["label"] == 1]
            zre = [z for z, r in zip(zs, d) if r["label"] == 0]
            print(f"  {cond:6} {v:9} {statistics.pstdev(zs):10.4f} {statistics.mean(zai):10.4f}"
                  f" {statistics.mean(zre):12.4f}")
        print()

    print("TEST 2 -- fit the threshold on ONE condition, apply it to the other two.\n")
    print("  Baseline is always the shipped 0.65. A transferred threshold that beats it on")
    print("  conditions it never saw is evidence for the calibration diagnosis; one that only")
    print("  wins at home is evidence for overfitting.\n")
    data = {c: V.load(c) for c in CONDS}
    if any(v is None for v in data.values()):
        print("  not all conditions scored yet.")
        return
    for src in CONDS:
        thr = best_thr(data[src], "B")
        held = [c for c in CONDS if c != src]
        print(f"  threshold fitted on {src:5} -> {thr:.4f}")
        for c in held:
            b0 = ba_at(data[c], "baseline", V.THRESH)
            b1 = ba_at(data[c], "B", thr)
            print(f"      HELD OUT {c:6} baseline@0.65 {b0:6.2f}%   B@{thr:.3f} {b1:6.2f}%"
                  f"   {b1-b0:+.2f}")
        home = ba_at(data[src], "B", thr)
        print(f"      (home     {src:6} {home:6.2f}%  -- fitted here, not evidence)\n")


if __name__ == "__main__":
    main()
