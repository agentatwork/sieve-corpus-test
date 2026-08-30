#!/usr/bin/env python3
"""Score one or more image FILES with the same rule score.py sweeps a directory with.

score.py takes `--set clean|web|hard` and walks `images/<set>/`; judging a bounty
submission means scoring an arbitrary path. Both call score_image() in score.py --
one copy of the rule, so a submission is scored by exactly the code that produced
the published false-positive numbers.

Metadata is never read. Sieve's extension short-circuits to 0.99 on a C2PA/"made
with AI" marker; that is a string test, not a measurement, and a submitter could
write it into a real photo's EXIF. Everything here is pixels only.

    python3 judge.py photo.jpg another.png --out results.json
    python3 judge.py subs/*.jpg --json
"""
import argparse, hashlib, json, os, sys

from PIL import Image

from score import load_ctx, score_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", help="write JSON records here")
    ap.add_argument("--json", action="store_true", help="also print JSON to stdout")
    a = ap.parse_args()

    ctx = load_ctx()
    out = []
    for path in a.files:
        rec = {"file": os.path.basename(path)}
        try:
            raw = open(path, "rb").read()
            rec["sha256"] = hashlib.sha256(raw).hexdigest()
            rec["bytes"] = len(raw)
            img = Image.open(path)
            rec["format"] = img.format
            img = img.convert("RGB")
            rec["w"], rec["h"] = img.size
            rec.update(score_image(img, ctx))
        except Exception as e:                       # a corrupt upload is a result, not a crash
            rec["error"] = f"{type(e).__name__}: {e}"
        out.append(rec)
        s = rec.get("score")
        print(f"{rec['file']:<40} "
              + (f"{s:.4f}  {'AI-FLAGGED' if s >= 0.5 else 'real'}"
                 f"{'  (tta)' if rec.get('tta') else ''}" if s is not None
                 else rec.get("skip") or rec.get("error")))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    if a.json:
        print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
