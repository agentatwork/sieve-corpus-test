#!/usr/bin/env python3
"""Post-hoc: is the PREREG2 result real, or an artifact of measuring disagreement in
probability space?

Read this as what it is. PREREG2 pre-registered Q1 and Q2, and on ft58s **both pass**, wide:
AUC(d) is 0.91 / 0.88 / 0.81 at clean / web / hard, beating the free control u by +0.33 / +0.26
/ +0.13. The pre-registered answer is "propose the two-view doubt signal". I am not withdrawing
that result and doubt.py is untouched -- the number a pre-registration produces is the number it
produces, and editing the instrument after seeing the output is how a pre-registration becomes
decoration.

But two things in the same output do not fit together:

  * At the 5% abstention operating point, the FREE near-threshold control catches MORE errors
    than disagreement at clean (6/19 vs 4/19) and exactly as many at web and hard. If d really
    carried +0.33 of AUC over u, it should not lose to u at the operating point anyone would
    ship.
  * 55-57% of images have d < 0.01.

Both are explained by the same thing, and it is my fault, in the design not the code.

PREREG2 defined d = |cal(z_std) - cal(z_nat)| -- a difference of *calibrated probabilities*.
cal() is a sigmoid, so it is steep near the threshold and flat in the tails. Two views that
disagree by a full 2.0 in logit space score d = 0.0013 if the image is confident (8.0 vs 6.0 ->
0.9998 vs 0.9985) and d = 0.10 if it is uncertain (0.1 vs -0.3 -> 0.60 vs 0.50). Identical
disagreement, 75x different d, decided entirely by where the image sits relative to the
threshold.

So d does not merely correlate with u. d is *built out of* u: a probability-space difference is
a logit-space difference multiplied by the local slope of the sigmoid, and that slope IS
proximity to the threshold. Q2 asked whether d beats u while d had u inside it.

The test that separates them is to measure disagreement where the sigmoid cannot compress it:

    d_logit = |z_std - z_nat|

which is the same disagreement with the threshold-proximity factor removed. If the second view
carries real independent doubt, d_logit should still beat u. If PREREG2's win was the sigmoid
slope wearing a disagreement costume, d_logit should collapse toward -- or below -- u.

And one control decides it outright. d_prob ~= sigma'(z) * d_logit, so take sigma'(z) = p(1-p)
on its own: a quantity computed from the FIRST view alone, structurally incapable of knowing
whether the two views agree on anything. If that reproduces d_prob's AUC, then d_prob's AUC was
never a measurement of the ensemble. This is the nonsense-selector control -- "the signal has no
function" and "the signal is absent" are different claims, and only a control separates them.

This file cannot make PREREG2 pass or fail. It reports which of those two the data says, and
that determines what I actually recommend upstream, which is not always what the pre-registered
rule alone would have had me recommend.
"""
import json, math, os

import variants as V

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ("clean", "web", "hard")


def auc(scores, labels):
    """Mann-Whitney U with ties at half. No sampling, no seed."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan"), len(pos)
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg)), len(pos)


def rows(cond):
    recs = [r for r in json.load(open(f"{HERE}/dual_{cond}.json"))
            if r.get("z_std") is not None and r.get("z_nat") is not None]
    out = []
    for r in recs:
        zs, zn = r["z_std"], r["z_nat"]
        cal = lambda z: 1.0 / (1.0 + math.exp(-(z / V.TEMP + V.BIAS)))
        shipped = V.apply(r, "baseline")
        out.append({
            "err": (shipped >= V.THRESH) != (r["label"] == 1),
            "d_prob": abs(cal(zs) - cal(zn)),          # PREREG2's signal
            "d_logit": abs(zs - zn),                   # the same disagreement, uncompressed
            "u": abs(cal(zs) - V.THRESH),              # the free control (smaller = doubtful)
            # The nonsense control. d_prob ~= sigma'(z) * d_logit, and sigma'(z) = p(1-p).
            # This is that slope with the disagreement deleted -- it is computed from the FIRST
            # view alone and cannot know anything about whether the views agree. If it scores
            # like d_prob, then d_prob's AUC was never about the second view.
            "slope": cal(zs) * (1 - cal(zs)),
        })
    return out


def main():
    print(f"MODEL {V.MODEL_VERSION}  bias {V.BIAS}  threshold {V.THRESH}")
    print("\nPOST-HOC. PREREG2 passed on d_prob; this asks whether d_prob is just u in disguise.")
    print("  AUC is of each signal predicting Sieve's error. u is negated so that, like the")
    print("  others, larger = more doubt. A signal at 0.50 is worthless.\n")
    print(f"  {'cond':6} {'n':>4} {'errs':>5} {'AUC(d_prob)':>12} {'AUC(d_logit)':>13}"
          f" {'AUC(u)':>8} {'AUC(slope)':>11} {'d_logit - u':>12}")

    verdict = {}
    for c in CONDS:
        rs = rows(c)
        errs = [r["err"] for r in rs]
        a_prob, n_err = auc([r["d_prob"] for r in rs], errs)
        a_log, _ = auc([r["d_logit"] for r in rs], errs)
        a_u, _ = auc([-r["u"] for r in rs], errs)
        a_sl, _ = auc([r["slope"] for r in rs], errs)
        verdict[c] = a_log - a_u
        print(f"  {c:6} {len(rs):4} {n_err:5} {a_prob:12.4f} {a_log:13.4f} {a_u:8.4f}"
              f" {a_sl:11.4f} {a_log - a_u:+12.4f}")

    print("\n  Q2 re-asked with the sigmoid slope removed (>= +0.03 at EVERY condition):")
    worst = min(verdict.values())
    for c in CONDS:
        print(f"    {c:6} {verdict[c]:+.4f}  {'clears' if verdict[c] >= 0.03 else 'FAILS'}")
    if worst >= 0.03:
        print("\n  The second view carries independent doubt. PREREG2's recommendation stands.")
    else:
        print(f"\n  It does not clear at every condition (worst {worst:+.4f}). PREREG2's Q2 win")
        print("  was the probability-space slope, not the second view. The honest upstream")
        print("  recommendation is the FREE one: an abstention band on the existing score.")
        print("  That needs no second inference, and it makes both of my pre-registrations")
        print("  irrelevant to the fix -- which PREREG2 said in advance I would say.")


if __name__ == "__main__":
    main()
