#!/usr/bin/env python3
"""What an abstention band on the EXISTING score would buy, at several widths.

This is the recommendation that survives confound.py: the second view's disagreement carries no
error information the first view's distance-to-threshold does not already carry for free, so the
useful lever is to stop pretending a score of 0.66 means the same thing as a score of 0.99.

No second inference, no new model, no extra pass over the pixels. Sieve already has the number.

The band is symmetric in *probability* around the 0.65 threshold and quoted by the coverage it
costs, because "abstain below 0.72" is not portable across recalibrations and "abstain on the 5%
of images nearest the threshold" is. Both columns are printed: the width is what you implement,
the coverage is what your users feel.

Read the lift column, not the caught column. Abstaining on 10% of images catches 10% of errors
by doing nothing at all -- lift is how much better than that this does. Lift 1.0 is worthless
however impressive the raw percentage looks.
"""
import json, math, os

import variants as V

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ("clean", "web", "hard")
RATES = (0.02, 0.05, 0.10, 0.15, 0.20)


def rows(cond):
    out = []
    for r in json.load(open(f"{HERE}/dual_{cond}.json")):
        if r.get("z_std") is None:
            continue
        p = V.apply(r, "baseline")
        out.append({"u": abs(p - V.THRESH), "p": p,
                    "err": (p >= V.THRESH) != (r["label"] == 1)})
    return out


def main():
    print(f"MODEL {V.MODEL_VERSION}  bias {V.BIAS}  threshold {V.THRESH}")
    print("\nAbstain on the images nearest the threshold. No second inference.\n")
    print(f"  {'cond':6} {'abstain':>8} {'n':>4} {'band (score)':>20} {'errors caught':>16} {'lift':>6}"
          f" {'errs left':>10}")
    for c in CONDS:
        rs = sorted(rows(c), key=lambda r: r["u"])
        tot = sum(r["err"] for r in rs)
        for rate in RATES:
            k = max(1, round(len(rs) * rate))
            top = rs[:k]
            caught = sum(r["err"] for r in top)
            lift = (caught / tot) / (k / len(rs)) if tot else float("nan")
            lo, hi = min(r["p"] for r in top), max(r["p"] for r in top)
            print(f"  {c:6} {rate:7.0%} {k:4} {lo:8.3f} - {hi:<9.3f} "
                  f"{caught:5}/{tot:<3} {caught/tot:6.1%} {lift:6.2f} {tot-caught:10}")
        print()

    print("  A caveat that belongs next to the table, not under it: the band edges above are")
    print("  fitted on the same 320 images they are scored on. The coverage column transfers")
    print("  (it is a rank rule); the score-range column is this corpus's, and a deployment")
    print("  should re-derive it. shrink.py is where I learned to say that -- a threshold fitted")
    print("  on one condition and applied to another lost on 5 of 6 transfers.")


if __name__ == "__main__":
    main()
