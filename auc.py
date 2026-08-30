#!/usr/bin/env python3
"""Secondary, threshold-free view of the same variants. NOT part of the decision rule.

PREREG.md commits the decision to balanced accuracy at 0.65. That is the operating point
Sieve actually ships, so it is the one that decides. But every image that flipped at `web`
landed within 0.06 of 0.65, which raises a fair question the pre-registered number cannot
answer: does variant A make the *ranking* worse, or does it just jitter three images across
an arbitrary line?

AUC answers that and nothing else. It is reported separately, and it does not overturn P4 --
mixing an operating point with a threshold-free statistic to get the verdict I wanted is the
exact move this file exists to avoid making silently.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variants as V


def auc(recs, variant):
    """Mann-Whitney U / (n_ai * n_real), with ties counted as half. No sampling, no seed."""
    ai = [V.apply(r, variant) for r in recs if r["label"] == 1]
    real = [V.apply(r, variant) for r in recs if r["label"] == 0]
    if not ai or not real:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in ai for b in real)
    return wins / (len(ai) * len(real))


def main():
    print("THRESHOLD-FREE (AUC). Secondary analysis -- does not decide anything.\n")
    print(f"  {'cond':6} {'baseline':>9} {'A':>9} {'delta':>8}   {'B':>9} {'delta':>8}")
    for cond in ("clean", "web", "hard"):
        d = V.load(cond)
        if d is None:
            continue
        a0, aA, aB = auc(d, "baseline"), auc(d, "A"), auc(d, "B")
        print(f"  {cond:6} {a0:9.4f} {aA:9.4f} {aA-a0:+8.4f}   {aB:9.4f} {aB-a0:+8.4f}")

    print("\n  Reading: AUC up while balanced accuracy at 0.65 is down means the variant ranks")
    print("  better but pushes a few images across the shipped threshold -- a calibration")
    print("  problem, not a discrimination one. AUC down means the second view genuinely")
    print("  carries less information for that condition, and no re-threshold rescues it.")


if __name__ == "__main__":
    main()
