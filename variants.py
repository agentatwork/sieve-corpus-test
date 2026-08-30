#!/usr/bin/env python3
"""Derive baseline / A / B from the stored dual logits and run the pre-registered checks.

Every variant sees bit-identical model output; the only thing that differs is the rule
deciding whether to average the second view. So any difference below is caused by the
gating rule and nothing else.

Two guards run first and are fatal, both named in PREREG.md before the sweep:
  G1  the recomputed baseline must reproduce the shipped scores_*.json
  G2  variant A must be identical to baseline for every image with min(w,h) >= 384
G2 is the one that matters -- it is what makes this a proposal about thumbnails rather
than a silent change to how Sieve scores everything.
"""
import json, math, os, sys
import sio

HERE = os.path.dirname(os.path.abspath(__file__))

# Read the calibration and gate from the manifest, never from constants typed here. These were
# hardcoded as 0.43 / 0.25 / 0.85 / 384 until upstream shipped ft58s with bias 0.30, at which
# point this harness would have gone on applying the old calibration to the new model's logits
# and reported the difference as a finding. The manifest is the single source of truth that
# score.py already uses; there is no second copy of it now.
_META = json.load(open(next(p for p in (f"{HERE}/model/model_manifest.json",
                                        "/tmp/sieve/extension/model_manifest.json")
                            if os.path.exists(p))))
_CAL, _TTA = _META.get("calibration", {}), _META.get("tta", {})
MODEL_VERSION = _META["version"]
BIAS, TEMP = _CAL.get("bias", 0.0), _CAL.get("temperature", 1.0)
BAND_LO, BAND_HI = _TTA["band_lo"], _TTA["band_hi"]
MIN_SIDE = _TTA.get("min_side") or _META["input_size"]
THRESH = 0.65
FACE = {"ffhq-256", "celeb-a-hq", "lfw", "idoc-mugshots-images"}

cal = lambda z: 1.0 / (1.0 + math.exp(-(z / TEMP + BIAS)))


def apply(rec, variant):
    """Return the calibrated score under a gating rule. Same two logits every time."""
    z_std, z_nat = rec["z_std"], rec["z_nat"]
    in_band = BAND_LO <= cal(z_std) <= BAND_HI
    big = min(rec["w"], rec["h"]) >= MIN_SIDE
    if variant == "baseline":
        use = in_band and big
    elif variant == "A":
        use = in_band
    elif variant == "B":
        use = True
    return cal((z_std + z_nat) / 2.0 if use else z_std)


def load(setname):
    """Return the dual-logit records, or None if the set is missing OR INCOMPLETE.

    dual.py checkpoints every 20 images and writes files in sorted order, and `ai__*`
    sorts before `real__*` -- so a partially written file is all-AI, which silently
    reports specificity 0.0% and a balanced accuracy of 26%. That is not a result, it
    is a half-finished scan, and it read exactly like a catastrophic regression the
    first time I looked at it. Refuse to score a set until its file list matches the
    shipped scores file for the same condition.
    """
    p = f"{HERE}/dual_{setname}.json"
    if not os.path.exists(p):
        return None
    recs = [r for r in json.load(open(p)) if "z_std" in r]
    ref = f"{HERE}/scores_{setname}.json"
    if os.path.exists(ref):
        want = {r["file"] for r in sio.records(ref)}
        have = {r["file"] for r in recs} | {r["file"] for r in json.load(open(p)) if "skip" in r}
        if want - have:
            print(f"  !! {setname}: INCOMPLETE, {len(want-have)} of {len(want)} images not yet"
                  f" scored -- excluded from every table below.")
            return None
    return recs


def metrics(recs, variant):
    ai = [r for r in recs if r["label"] == 1]
    real = [r for r in recs if r["label"] == 0]
    faces = [r for r in real if r["source"] in FACE]
    rec = 100.0 * sum(apply(r, variant) >= THRESH for r in ai) / max(len(ai), 1)
    spec = 100.0 * sum(apply(r, variant) < THRESH for r in real) / max(len(real), 1)
    fspec = 100.0 * sum(apply(r, variant) < THRESH for r in faces) / max(len(faces), 1)
    return {"n_ai": len(ai), "n_real": len(real), "n_face": len(faces),
            "recall": rec, "spec": spec, "ba": (rec + spec) / 2, "face_spec": fspec}


def guards():
    ok = True
    # G0 runs first and hard-fails, because every guard below it compares numbers across
    # files and is meaningless if those files came from different models. G1 in particular
    # would report an ft44s-vs-ft58s calibration shift (bias 0.43 -> 0.30) as a reproduction
    # failure in this harness, which is precisely the wrong conclusion and the one I drew.
    shipped = [f"{HERE}/scores_{c}.json" for c in ("clean", "web", "hard")]
    have = [p for p in shipped if os.path.exists(p)]
    if have:
        v = sio.require_version(have, want=MODEL_VERSION)   # raises; does not warn
        print(f"  G0 every reference score file is {v}: OK ")

    for cond in ("clean", "web", "hard"):
        d, shipped = load(cond), f"{HERE}/scores_{cond}.json"
        if d is None or not os.path.exists(shipped):
            continue
        old = {r["file"]: r for r in sio.records(shipped) if "score" in r}
        worst = 0.0
        for r in d:
            if r["file"] in old:
                worst = max(worst, abs(apply(r, "baseline") - old[r["file"]]["score"]))
        flag = "OK " if worst < 1e-9 else "FAIL"
        ok &= worst < 1e-9
        print(f"  G1 {cond:5} baseline reproduces shipped scores: {flag} (max |delta| {worst:.2e})")
    for cond in ("clean", "web", "hard"):
        d = load(cond)
        if d is None:
            continue
        big = [r for r in d if min(r["w"], r["h"]) >= MIN_SIDE]
        bad = [r for r in big if abs(apply(r, "A") - apply(r, "baseline")) > 0]
        flag = "OK " if not bad else "FAIL"
        ok &= not bad
        print(f"  G2 {cond:5} variant A untouched for the {len(big):3} images >=384px: {flag}")
    return ok


def flips(recs, v1, v2):
    out = []
    for r in recs:
        a, b = apply(r, v1), apply(r, v2)
        if (a >= THRESH) != (b >= THRESH):
            correct = (b >= THRESH) == (r["label"] == 1)
            out.append((r, a, b, correct))
    return out


def main():
    print(f"MODEL {MODEL_VERSION}   bias {BIAS}  temp {TEMP}  band {BAND_LO}-{BAND_HI}"
          f"  min_side {MIN_SIDE}  threshold {THRESH}")
    print("(Sieve shipped 3 models in 9 days. Any table below is about this one only.)\n")
    print("GUARDS")
    if not guards():
        print("\n  a guard failed -- results below are not trustworthy, stopping.")
        sys.exit(1)

    print("\nPOOLED, threshold 0.65")
    print(f"  {'cond':6} {'variant':9} {'recall':>7} {'spec':>7} {'BA':>7} {'face spec':>10}")
    base = {}
    for cond in ("clean", "web", "hard"):
        d = load(cond)
        if d is None:
            continue
        for v in ("baseline", "A", "B"):
            m = metrics(d, v)
            if v == "baseline":
                base[cond] = m
            d_ba = "" if v == "baseline" else f"  ({m['ba']-base[cond]['ba']:+.2f})"
            d_fs = "" if v == "baseline" else f" ({m['face_spec']-base[cond]['face_spec']:+.1f})"
            print(f"  {cond:6} {v:9} {m['recall']:6.1f}% {m['spec']:6.1f}% {m['ba']:6.2f}%{d_ba:9}"
                  f" {m['face_spec']:6.1f}%{d_fs}")

    print(f"\nSTYLEGAN FACES (n=24, AI, detected at >= {THRESH})")
    for cond in ("sg_clean", "sg_web", "sg_hard"):
        d = load(cond)
        if d is None:
            continue
        row = "  " + f"{cond:9}"
        for v in ("baseline", "A", "B"):
            row += f"  {v}={sum(apply(r,v)>=THRESH for r in d):2}/{len(d)}"
        print(row)

    g = load("graphics")
    if g:
        print(f"\nHUMAN-MADE NON-PHOTOGRAPHS (n={len(g)}, any hit at >= {THRESH} is a false accusation)")
        for v in ("baseline", "A", "B"):
            print(f"  {v:9} false positives = {sum(apply(r,v)>=THRESH for r in g)}"
                  f"   max score = {max(apply(r,v) for r in g):.3f}")

    for v in ("A", "B"):
        d = load("hard")
        if not d:
            break
        f = flips(d, "baseline", v)
        print(f"\nFLIPS AT hard, baseline -> {v}: {len(f)} images "
              f"({sum(1 for x in f if x[3])} corrected, {sum(1 for x in f if not x[3])} broken)")
        for r, a, b, corr in sorted(f, key=lambda x: -abs(x[2] - x[1])):
            print(f"  {'FIXED ' if corr else 'BROKE '} {'AI ' if r['label'] else 'REAL'}"
                  f" {r['source']:30} {r['w']}x{r['h']}  {a:.3f} -> {b:.3f}")

    print("\nPRE-REGISTERED DECISION (variant A)")
    h, hb = load("hard"), base.get("hard")
    if h and hb:
        ma = metrics(h, "A")
        p1 = ma["ba"] - hb["ba"] >= 0.0
        p2 = ma["face_spec"] - hb["face_spec"] >= 2.5
        sg = load("sg_hard")
        p3 = True
        if sg:
            p3 = sum(apply(r, "A") >= THRESH for r in sg) >= sum(apply(r, "baseline") >= THRESH for r in sg) - 1
        # P4 must not pass by absence: an unscored condition cannot vouch for the variant.
        have = [c for c in ("clean", "web") if load(c)]
        p4 = len(have) == 2 and all(metrics(load(c), "A")["ba"] - base[c]["ba"] >= -0.5 for c in have)
        if len(have) < 2:
            print(f"  P4 cannot be evaluated yet: {2-len(have)} of clean/web still unscored.")
        for k, v, why in (("P1", p1, f"hard BA delta {ma['ba']-hb['ba']:+.2f} >= 0"),
                          ("P2", p2, f"hard face spec delta {ma['face_spec']-hb['face_spec']:+.1f} >= +2.5"),
                          ("P3", p3, "stylegan hard recall loses <= 1"),
                          ("P4", p4, "clean and web BA each >= -0.5")):
            print(f"  {k} {'PASS' if v else 'FAIL'}  {why}")
        print("\n  =>", "PROPOSE variant A upstream." if (p1 and p2 and p3 and p4)
              else "DO NOT propose. Publish the negative result.")


if __name__ == "__main__":
    main()
