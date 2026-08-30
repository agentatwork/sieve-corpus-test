#!/usr/bin/env python3
"""What the ft8 legacy-generator round actually bought, per cluster, per condition.

Upstream's v0.12.0 note reports large held-out gains on the GenImage cohort (BigGAN 50 -> 90%,
GLIDE 80 -> 89%, ADM 45 -> 65%, VQDM 79 -> 93%). This repository has the same two models scored
over the same 320 images at three delivery conditions, so it can ask a question a held-out slice
at one condition cannot: did the gain survive the delivery path?

ft44s scores are the quarantined v0.10-ft44s/ run; they predate sio.py and carry no version
stamp, which is why they are read positionally here and labelled by their directory rather than
by a field. That is a weaker guarantee than G0 gives the root files, and it is the reason
v0.10-ft44s/ is frozen -- nothing new is ever written into it.
"""
import os, sio

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ("clean", "web", "hard")
THRESH = 0.65


def recall(path):
    """cluster -> (detected, total) over AI images only."""
    out = {}
    for r in sio.records(path):
        if r["label"] != 1 or r.get("score") is None:
            continue
        d, t = out.get(r["source"], (0, 0))
        out[r["source"]] = (d + (r["score"] >= THRESH), t + 1)
    return out


def main():
    old = {c: recall(f"{HERE}/v0.10-ft44s/scores_{c}.json") for c in CONDS}
    new = {c: recall(f"{HERE}/scores_{c}.json") for c in CONDS}
    clusters = sorted(new["clean"], key=lambda k: min(
        100 * new[c][k][0] / new[c][k][1] for c in CONDS))

    print(f"Recall %, 10 images per cluster, threshold {THRESH}.  ft44s (v0.10) -> ft58s (v0.12)\n")
    print(f"  {'cluster':34} " + "  ".join(f"{c:^16}" for c in CONDS))
    for k in clusters:
        cells = []
        for c in CONDS:
            o = 100 * old[c].get(k, (0, 0))[0] / max(1, old[c].get(k, (0, 1))[1])
            n = 100 * new[c][k][0] / new[c][k][1]
            cells.append(f"{o:5.0f} ->{n:4.0f} {n - o:+4.0f}")
        print(f"  {k:34} " + "  ".join(cells))

    print("\n  Cohort totals (the four clusters upstream's ft8 round targeted):")
    coh = [k for k in clusters if k.startswith("GenImage_") and
           any(g in k for g in ("ADM", "BigGAN", "glide", "VQDM"))]
    for c in CONDS:
        od = sum(old[c].get(k, (0, 0))[0] for k in coh); ot = sum(old[c].get(k, (0, 1))[1] for k in coh)
        nd = sum(new[c][k][0] for k in coh); nt = sum(new[c][k][1] for k in coh)
        print(f"    {c:6} {100*od/ot:5.1f} -> {100*nd/nt:5.1f}  ({od}/{ot} -> {nd}/{nt})")


if __name__ == "__main__":
    main()
