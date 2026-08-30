#!/usr/bin/env python3
"""Turn Sieve's scores on the shared 320-image corpus into the tables that say
something actionable.

Reported at the fixed 0.65 operating point, both classes always separate, and
per source cluster — because a single balanced-accuracy number hides the only
thing a detector author can act on, which is *which* generator or *which* kind
of real photograph it is wrong about.

  python3 analyze.py
"""
import json, os, math, collections
import sio

HERE = os.path.dirname(os.path.abspath(__file__))
TH = 0.65
SETS = ["clean", "web", "hard"]


def load(s):
    return sio.records(os.path.join(HERE, f"scores_{s}.json"))


def wilson(k, n, z=1.96):
    """Wilson interval — a normal approximation on 10 trials is nonsense."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def headline(rows):
    ai = [r for r in rows if r["label"] == 1 and "score" in r]
    re_ = [r for r in rows if r["label"] == 0 and "score" in r]
    rec = sum(r["score"] >= TH for r in ai)
    spec = sum(r["score"] < TH for r in re_)
    return {
        "n_ai": len(ai), "n_real": len(re_),
        "recall": 100 * rec / len(ai), "spec": 100 * spec / len(re_),
        "ba": 50 * (rec / len(ai) + spec / len(re_)),
        "rec_ci": wilson(rec, len(ai)), "spec_ci": wilson(spec, len(re_)),
        "tta": sum(1 for r in rows if r.get("tta")),
        "skipped": sum(1 for r in rows if "skip" in r),
    }


def main():
    data = {s: load(s) for s in SETS}

    print("=" * 78)
    print("HEADLINE — balanced accuracy at the fixed 0.65 threshold")
    print("=" * 78)
    print(f"{'condition':10s} {'BA':>7s} {'recall':>8s} {'95% CI':>16s} {'specificity':>12s} "
          f"{'95% CI':>16s} {'TTA':>5s}")
    for s in SETS:
        h = headline(data[s])
        print(f"{s:10s} {h['ba']:7.1f} {h['recall']:8.1f} "
              f"[{h['rec_ci'][0]:5.1f},{h['rec_ci'][1]:5.1f}] {h['spec']:12.1f} "
              f"[{h['spec_ci'][0]:5.1f},{h['spec_ci'][1]:5.1f}] {h['tta']:5d}")
    h = headline(data["clean"])
    print(f"\nn = {h['n_ai']} AI (18 generator clusters) + {h['n_real']} real "
          f"(14 source clusters); skipped as degenerate: "
          f"{ {s: headline(data[s])['skipped'] for s in SETS} }")

    print("\n" + "=" * 78)
    print("PER-GENERATOR RECALL (10 images each) — where the misses are")
    print("=" * 78)
    gens = sorted({r["source"] for r in data["clean"] if r["label"] == 1})
    print(f"{'generator':34s} {'clean':>8s} {'web':>8s} {'hard':>8s}   {'worst':>6s}")
    rows = []
    for g in gens:
        vals = []
        for s in SETS:
            rs = [r for r in data[s] if r["source"] == g and "score" in r]
            vals.append(100 * sum(r["score"] >= TH for r in rs) / max(1, len(rs)))
        rows.append((min(vals), g, vals))
    for worst, g, vals in sorted(rows):
        mark = "   <-- MISS" if worst <= 50 else ("   <- weak" if worst < 80 else "")
        print(f"{g:34s} {vals[0]:8.0f} {vals[1]:8.0f} {vals[2]:8.0f}   {worst:6.0f}{mark}")

    print("\n" + "=" * 78)
    print("PER-REAL-SOURCE SPECIFICITY (10 images each) — where the false alarms are")
    print("=" * 78)
    reals = sorted({r["source"] for r in data["clean"] if r["label"] == 0})
    print(f"{'real source':34s} {'clean':>8s} {'web':>8s} {'hard':>8s}   {'worst':>6s}")
    rows = []
    for g in reals:
        vals = []
        for s in SETS:
            rs = [r for r in data[s] if r["source"] == g and "score" in r]
            vals.append(100 * sum(r["score"] < TH for r in rs) / max(1, len(rs)))
        rows.append((min(vals), g, vals))
    for worst, g, vals in sorted(rows):
        mark = "   <-- FALSE ALARMS" if worst <= 80 else ""
        print(f"{g:34s} {vals[0]:8.0f} {vals[1]:8.0f} {vals[2]:8.0f}   {worst:6.0f}{mark}")

    print("\n" + "=" * 78)
    print("HOW IT FAILS — does degradation make it wrong, or make it quiet?")
    print("=" * 78)
    for s in SETS:
        h = headline(data[s])
        print(f"{s:6s} recall {h['recall']:5.1f}  specificity {h['spec']:5.1f}")
    c, hd = headline(data["clean"]), headline(data["hard"])
    print(f"\nclean -> hard: recall {c['recall']:.1f} -> {hd['recall']:.1f} "
          f"({hd['recall']-c['recall']:+.1f}), specificity {c['spec']:.1f} -> {hd['spec']:.1f} "
          f"({hd['spec']-c['spec']:+.1f})")

    print("\n" + "=" * 78)
    print("THE BAND — how many land in the amber 'unsure' zone (0.50-0.65)")
    print("=" * 78)
    for s in SETS:
        rs = [r for r in data[s] if "score" in r]
        band = [r for r in rs if 0.50 <= r["score"] < TH]
        ai_band = sum(1 for r in band if r["label"] == 1)
        print(f"{s:6s} {len(band):3d}/{len(rs)} unsure  ({ai_band} AI, {len(band)-ai_band} real)")

    print("\n" + "=" * 78)
    print("SIZE — my corpus is mostly small, which is the thumbnail regime")
    print("=" * 78)
    sizes = collections.Counter()
    for r in data["clean"]:
        m = min(r["w"], r["h"])
        sizes["<384 (no TTA possible)" if m < 384 else ">=384"] += 1
    print(dict(sizes))

    # ---- targeted probe: StyleGAN faces ------------------------------------
    # BigGAN scoring 0% on the main corpus says the model has no handle on GAN
    # artefacts. The GAN image that actually matters on the web is not BigGAN,
    # it is a 1024x1024 StyleGAN face used as a profile picture, so that is the
    # test worth running: 24 fresh ones straight from thispersondoesnotexist.
    sg = {}
    for s in SETS:
        p = os.path.join(HERE, f"scores_sg_{s}.json")
        if os.path.exists(p):
            sg[s] = sio.records(p)
    if sg:
        print("\n" + "=" * 78)
        print("STYLEGAN FACES (thispersondoesnotexist, fetched fresh, 1024x1024)")
        print("=" * 78)
        for s, rows in sg.items():
            r = [x for x in rows if "score" in x]
            hit = sum(x["score"] >= TH for x in r)
            band = sum(0.50 <= x["score"] < TH for x in r)
            med = sorted(x["score"] for x in r)[len(r) // 2]
            lo, hi = wilson(hit, len(r))
            print(f"  {s:6s} detected {hit:2d}/{len(r)}  ({100*hit/len(r):5.1f}% "
                  f"[{lo:.1f},{hi:.1f}])  amber-unsure {band}  median score {med:.4f}")

    # ---- targeted probe: non-photographs -----------------------------------
    gp = os.path.join(HERE, "scores_graphics.json")
    if os.path.exists(gp):
        print("\n" + "=" * 78)
        print("NON-PHOTOGRAPHS — brand graphics, UI screenshots, charts, posters")
        print("(all human-made, none AI; a score >= 0.65 here is a false accusation)")
        print("=" * 78)
        for r in sorted(sio.records(gp), key=lambda x: -x.get("score", -1)):
            name = r["file"].split("__")[-1].replace(".jpg", "")
            if "skip" in r:
                print(f"  {name:24s}  SKIPPED SILENTLY ({r['skip']}) — no badge, no verdict")
            else:
                flag = "  <-- FLAGGED AS AI" if r["score"] >= TH else (
                       "  <- amber unsure" if r["score"] >= 0.5 else "")
                print(f"  {name:24s}  {r['score']:.4f}{flag}")

    out = {s: headline(data[s]) for s in SETS}
    if sg:
        out["stylegan"] = {s: {"n": len([x for x in rows if "score" in x]),
                               "detected": sum(x.get("score", 0) >= TH for x in rows)}
                           for s, rows in sg.items()}
    json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
