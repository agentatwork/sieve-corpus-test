#!/usr/bin/env python3
"""Check that the degenerate_reason rewrite changed nothing except how much memory it uses.

`degenerate_reason` used to cast the whole image to float32 and then keep about one row in
every 128. It now slices first and casts the slice. The argument that this is safe is an
argument about numpy: uint8 -> float32 is lossless, and an elementwise cast commutes with a
slice, so the array that reaches the luminance line is bit-identical either way.

That argument is correct and it is still only an argument. This script is the measurement.

It checks three separate things, and they are separate on purpose:

  A. Function parity.   The published implementation and the current one are run against
                        every image, and every returned reason must match. The old function
                        is not retyped here -- it is loaded out of git history, so what gets
                        compared is the code that produced the published scores_*.json rather
                        than my recollection of it. [[one-rule-two-copies]]

  B. The numeric claim. Cast-then-slice and slice-then-cast are computed side by side and the
                        maximum absolute difference is recorded, on the luminance plane and on
                        its variance and lag-1 correlation. A is a pass/fail on a three-way
                        branch and would look identical if the numbers moved a little and
                        never crossed a threshold; B is the quantity itself.

  C. The memory.        Why the change was made at all. Reported as the peak allocation each
                        version needs for the sampled rows, on the largest image present.

None of it needs the model file, so this runs on a machine that does not have the 43 MB of
weights. It does import score.py, which imports onnxruntime at module scope, so onnxruntime
still has to be installed. It also reads the baseline out of git history, so a `--depth 1`
clone will not have the commit; it says so and exits rather than guessing.

  python3 parity.py                              # images/ beside this file
  python3 parity.py --images /path/to/images     # elsewhere
"""
import argparse, importlib.util, json, os, subprocess, sys, tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# The commit that published scores_clean.json / scores_web.json / scores_hard.json, and so the
# definition of degenerate_reason those files were produced by. Pinned rather than expressed as
# HEAD~n: this file will outlive the commit that adds it, and "one before the current tip" stops
# meaning the published version the moment anything else lands.
BASELINE = "537f834e961866c28c265d8465a991c3d4ee0201"


def load_baseline(rev):
    """Import degenerate_reason as of `rev`, straight out of the object database."""
    src = subprocess.run(["git", "-C", HERE, "show", f"{rev}:score.py"],
                         capture_output=True, text=True)
    if src.returncode:
        sys.exit(f"cannot read score.py at {rev}: {src.stderr.strip()}")
    fd, path = tempfile.mkstemp(suffix="_baseline.py", dir=tempfile.gettempdir())
    with os.fdopen(fd, "w") as fh:
        fh.write(src.stdout)
    spec = importlib.util.spec_from_file_location("score_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.unlink(path)
    return mod.degenerate_reason


def lum_both_ways(img):
    """The two orderings, computed explicitly, with nothing else between them.

    This deliberately re-derives the two lines rather than calling either implementation: the
    point is to compare the orderings, and calling the functions would only tell me again what
    check A already tells me.
    """
    px = np.asarray(img)
    h = px.shape[0]
    sy = max(1, h >> 7)

    old_rows = px.astype(np.float32)[::sy]              # cast the image, then slice
    new_rows = px[::sy].astype(np.float32)              # slice, then cast the rows

    def stats(rows):
        lum = 0.299 * rows[:, :, 0] + 0.587 * rows[:, :, 1] + 0.114 * rows[:, :, 2]
        prev, cur = lum[:, :-1].ravel(), lum[:, 1:].ravel()
        if prev.size == 0:
            return lum, float(lum.var()), None
        cov = (prev * cur).mean() - prev.mean() * cur.mean()
        denom = np.sqrt(max(prev.var() * cur.var(), 1e-9))
        return lum, float(lum.var()), float(cov / denom)

    lo, vo, co = stats(old_rows)
    ln, vn, cn = stats(new_rows)
    corr_d = 0.0 if co is None or cn is None else abs(co - cn)
    # Peak float32 bytes each ordering has to hold to produce `rows`: the old one materialises
    # the whole image, the new one only the sampled rows.
    return (float(np.abs(lo - ln).max()), abs(vo - vn), corr_d,
            px.size * 4, new_rows.size * 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(HERE, "images"))
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--out", default=os.path.join(HERE, "parity.json"))
    a = ap.parse_args()

    import score                                        # the current implementation
    old = load_baseline(a.baseline)

    files = sorted(os.path.join(r, f)
                   for r, _, fs in os.walk(a.images) for f in fs)
    if not files:
        sys.exit(f"no images under {a.images} -- the corpus is not published with this repo; "
                 f"pass --images at the directory scored by score.py")

    dis, worst = [], dict(lum=0.0, var=0.0, corr=0.0)
    reasons, biggest = {}, dict(px=0)
    for i, p in enumerate(files):
        img = Image.open(p).convert("RGB")              # exactly what score.py hands it
        w, h = img.size
        if min(w, h) < 32:                              # score.py never calls it on these
            continue
        ro, rn = old(img), score.degenerate_reason(img)
        reasons[str(rn)] = reasons.get(str(rn), 0) + 1
        if ro != rn:
            dis.append(dict(file=os.path.relpath(p, a.images), old=ro, new=rn))
        dl, dv, dc, old_b, new_b = lum_both_ways(img)
        worst["lum"] = max(worst["lum"], dl)
        worst["var"] = max(worst["var"], dv)
        worst["corr"] = max(worst["corr"], dc)
        if w * h > biggest["px"]:
            biggest = dict(px=w * h, w=w, h=h,
                           file=os.path.relpath(p, a.images),
                           old_bytes=int(old_b), new_bytes=int(new_b))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}", file=sys.stderr, flush=True)

    out = dict(
        baseline=a.baseline,
        # Images actually put through the function, which is not len(files): score.py skips
        # anything under 32px before it ever calls degenerate_reason, so this must count the
        # calls rather than the directory.
        n_images=sum(reasons.values()),
        n_files_seen=len(files),
        disagreements=dis,
        # Zero is the only passing value. It is written out rather than asserted so that a
        # reader can tell "ran and found nothing" from "never ran". [[absent-function-vs-false]]
        max_abs_delta=worst,
        reasons=reasons,
        largest_image=biggest,
        # The 9248x6936 case is arithmetic, not a measurement: no image that size is in this
        # corpus, which is exactly why the bug was invisible here. 9248*6936*3*4 bytes.
        motivating_case=dict(w=9248, h=6936,
                             old_bytes=9248 * 6936 * 3 * 4,
                             new_bytes=((6936 + (6936 >> 7) - 1) // (6936 >> 7)) * 9248 * 3 * 4),
    )
    json.dump(out, open(a.out, "w"), indent=1)

    ok = not dis and max(worst.values()) == 0.0
    print(f"{out['n_images']} images  |  reason disagreements: {len(dis)}  |  "
          f"max |delta| lum {worst['lum']:.17g} var {worst['var']:.17g} "
          f"corr {worst['corr']:.17g}")
    print(f"largest here {biggest['w']}x{biggest['h']}: "
          f"{biggest['old_bytes']/1e6:.1f} MB -> {biggest['new_bytes']/1e6:.1f} MB;  "
          f"9248x6936 would be {out['motivating_case']['old_bytes']/1e6:.0f} MB -> "
          f"{out['motivating_case']['new_bytes']/1e6:.1f} MB")
    print(("PASS" if ok else "FAIL") + f"  -> {os.path.relpath(a.out, HERE)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
