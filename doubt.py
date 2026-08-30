#!/usr/bin/env python3
"""Does the second view disagree with the first where Sieve is wrong? (PREREG2.md)

Answers the question the maintainer put to plotarmordev in issue #42 -- whether the views
disagree *within themselves* on real web images, or all flip together -- and tests whether
that disagreement is worth two inferences as a doubt signal.

Every disagreement number is printed beside the free control: distance from the threshold,
computed from the view Sieve already has. If the control wins, the second view is not worth
paying for and the recommendation is an abstention band on the existing score.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variants as V

CONDS = ("clean", "web", "hard")


def auc(scores, labels):
    """P(score of a positive > score of a negative), ties at half. Positives are errors."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan"), len(pos)
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg)), len(pos)


def signals(recs):
    """Per image: disagreement d, free control u, and whether the shipped verdict is wrong."""
    out = []
    for r in recs:
        s_std, s_nat = V.cal(r["z_std"]), V.cal(r["z_nat"])
        shipped = V.apply(r, "baseline")
        out.append({
            "d": abs(s_std - s_nat),
            "u": -abs(shipped - V.THRESH),      # negated: larger = closer to the line = doubt
            "err": (shipped >= V.THRESH) != (r["label"] == 1),
            "label": r["label"], "source": r["source"],
            "w": r["w"], "h": r["h"], "s_std": s_std, "s_nat": s_nat,
        })
    return out


def main():
    print(f"MODEL {V.MODEL_VERSION}  bias {V.BIAS}  threshold {V.THRESH}\n")

    print("Q1/Q2 -- can view disagreement predict Sieve's errors, better than free doubt?\n")
    print(f"  {'cond':6} {'n':>4} {'errs':>5} {'AUC(d)':>8} {'AUC(u) free':>12} {'d - u':>8}"
          f" {'ties d<.01':>11}")
    res = {}
    for c in CONDS:
        recs = V.load(c)
        if recs is None:
            continue
        sig = signals(recs)
        errs = [s["err"] for s in sig]
        ad, n_err = auc([s["d"] for s in sig], errs)
        au, _ = auc([s["u"] for s in sig], errs)
        ties = 100.0 * sum(s["d"] < 0.01 for s in sig) / len(sig)
        res[c] = {"ad": ad, "au": au, "n_err": n_err, "sig": sig}
        print(f"  {c:6} {len(sig):4} {n_err:5} {ad:8.4f} {au:12.4f} {ad-au:+8.4f} {ties:10.1f}%")

    if len(res) == 3:
        q1 = res["hard"]["ad"] >= 0.70 and res["clean"]["ad"] >= 0.65 and res["web"]["ad"] >= 0.65
        q2 = all(res[c]["ad"] - res[c]["au"] >= 0.03 for c in CONDS)
        print(f"\n  Q1 {'PASS' if q1 else 'FAIL'}  AUC(d) >= 0.70 hard, >= 0.65 clean/web")
        print(f"  Q2 {'PASS' if q2 else 'FAIL'}  AUC(d) - AUC(u) >= +0.03 at every condition")
        if q1 and q2:
            print("\n  => Propose the two-view doubt signal.")
        elif q1:
            print("\n  => Do NOT propose the second view. Disagreement does predict error, but not"
                  "\n     better than distance-from-threshold, which is free. Propose an abstention"
                  "\n     band on the existing score instead -- zero extra inference.")
        else:
            print("\n  => The second view carries no usable doubt signal. Report the negative.")

    print("\nSECONDARY -- abstain on the most-disagreeing 5%, how many errors does that catch?")
    print("  (reported for both signals; lift 1.0 = no better than abstaining at random)\n")
    for c in CONDS:
        if c not in res:
            continue
        sig, tot = res[c]["sig"], res[c]["n_err"]
        if not tot:
            continue
        k = max(1, len(sig) // 20)
        line = f"  {c:6}"
        for key, name in (("d", "disagree"), ("u", "near-thr")):
            top = sorted(sig, key=lambda s: -s[key])[:k]
            caught = sum(s["err"] for s in top)
            line += f"   {name} {caught:2}/{tot:2} errors ({100.0*caught/tot:4.1f}%, lift {caught/tot/(k/len(sig)):.2f})"
        print(line)

    print("\nDOES IT FLIP TOGETHER? -- the maintainer's actual question, on real images only")
    print("  A view pair that always agrees cannot be an ensemble OR a doubt signal.\n")
    for c in CONDS:
        if c not in res:
            continue
        real = [s for s in res[c]["sig"] if s["label"] == 0]
        split = [s for s in real if (s["s_std"] >= V.THRESH) != (s["s_nat"] >= V.THRESH)]
        both_wrong = [s for s in real if s["s_std"] >= V.THRESH and s["s_nat"] >= V.THRESH]
        print(f"  {c:6} of {len(real)} real images: views split on {len(split):3}"
              f"  |  both call it AI (flipped together) {len(both_wrong):3}")


if __name__ == "__main__":
    main()
