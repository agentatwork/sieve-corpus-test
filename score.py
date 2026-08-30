#!/usr/bin/env python3
"""Score images exactly the way the shipped Sieve extension does, in Python.

Not a re-implementation from the README: every step below is transcribed from
`extension/src/offscreen.js` at the commit under test, and the constants come
from the extension's own `model_manifest.json`. Sieve's browser resampler is a
hand-written Pillow-exact triangle filter, so PIL's own BILINEAR is the
reference it is chasing — which is why this can be trusted to stand in for the
browser on a machine that cannot run WebGPU.

  metadata sniff -> 0.99                        (handled separately, see meta_probe.mjs)
  reject <32px, flat (luma var < 4), noise (adjacent-pixel corr < 0.15)
  resize shorter side -> 440 (BILINEAR), centre-crop 384, ImageNet norm
  zStd = logit
  if 0.25 <= sigmoid(zStd + bias) <= 0.85 and min(w,h) >= 384:
      zNative = logit(native-resolution centre crop 384)
      z = (zStd + zNative) / 2          <- averaged in LOGIT space, not probability
  score = sigmoid(z + bias)

The averaging space is load-bearing and easy to get wrong in the flattering
direction; it is logits.

  python3 score.py --set clean --out scores_clean.json
"""
import argparse, hashlib, json, os, sys, time

import sio

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# The checkout lives in /tmp, which does not survive a reboot -- and this scorer has to keep
# judging bounty submissions until 7 September. Prefer the copy kept beside the code.
MANIFEST = next(p for p in (f"{HERE}/model/model_manifest.json",
                            "/tmp/sieve/extension/model_manifest.json") if os.path.exists(p))
# Which weights to load is the manifest's decision, not a constant here. Sieve shipped three
# models in nine days (ft44s -> ft5s -> ft58s) and changed the calibration bias 0.43 -> 0.30
# on the way; a hardcoded filename kept happily scoring against the old one. The sha256 check
# in load_ctx() is what makes this safe -- name and weights have to agree with the manifest.
_WANT = os.path.basename(json.load(open(MANIFEST))["model_url"])
MODEL = next(p for p in (f"{HERE}/model/{_WANT}",
                         f"/tmp/sieve/dev_model/{_WANT}") if os.path.exists(p))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img):
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def preprocess(img, S, C):
    w, h = img.size
    scale = S / min(w, h)
    rw, rh = max(C, round(w * scale)), max(C, round(h * scale))
    r = img.resize((rw, rh), Image.BILINEAR)
    x0, y0 = (rw - C) // 2, (rh - C) // 2
    return to_tensor(r.crop((x0, y0, x0 + C, y0 + C)))


def preprocess_native(img, C):
    w, h = img.size
    x0, y0 = (w - C) // 2, (h - C) // 2
    return to_tensor(img.crop((x0, y0, x0 + C, y0 + C)))


def degenerate_reason(img):
    """offscreen.js degenerateReason(), same sampling stride and thresholds.

    The subsample happens before the float cast, not after. Casting first and slicing second
    gives bit-identical numbers -- uint8 to float32 is lossless and an elementwise cast commutes
    with a slice -- but it materialises the whole image at 4 bytes per channel to keep about one
    row in every 128. On a 9248x6936 Commons original that is 770 MB for 129 rows, which is what
    killed a scan on a 2 GB box. offscreen.js never had this problem: it reads a canvas as a
    Uint8ClampedArray, so the float32 blowup was an artifact of transcribing it into numpy.
    """
    a = np.asarray(img)                       # uint8, as the browser sees it
    h, w, _ = a.shape
    sy = max(1, h >> 7)
    rows = a[::sy].astype(np.float32)
    lum = 0.299 * rows[:, :, 0] + 0.587 * rows[:, :, 1] + 0.114 * rows[:, :, 2]
    if lum.var() < 4:
        return "flat"
    prev, cur = lum[:, :-1].ravel(), lum[:, 1:].ravel()
    if prev.size == 0:
        return None
    cov = (prev * cur).mean() - prev.mean() * cur.mean()
    denom = np.sqrt(max(prev.var() * cur.var(), 1e-9))
    if cov / denom < 0.15:
        return "noise"
    return None


def load_ctx():
    """Model session + the manifest constants, in one object both entry points share."""
    meta = json.load(open(MANIFEST))
    # the bounty text names this hash, and the results page says it is checked every run;
    # a claim like that has to be computed, not asserted in prose
    h = hashlib.sha256(open(MODEL, "rb").read()).hexdigest()
    if h != meta["sha256"]:
        raise SystemExit(f"model sha256 {h} != manifest {meta['sha256']} ({MODEL})")
    cal = meta["calibration"]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1        # one core; more threads only thrash
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    temp = cal.get("temperature", 1.0) or 1.0
    return {"S": meta["resize_shorter_side"], "C": meta["input_size"],
            "bias": cal.get("bias", 0.0), "temp": temp, "tta": meta.get("tta", {}),
            "logit": lambda t: float(sess.run(None, {iname: t})[0].ravel()[0])}


def score_image(img, ctx):
    """The scoring rule itself. One copy: score.py's sweep and judge.py both call this.

    Returns the fields that go in a record -- either {"skip": reason} or the
    logits and the calibrated score.
    """
    S, C, tta_cfg = ctx["S"], ctx["C"], ctx["tta"]
    sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
    calibrate = lambda z: float(sigmoid(z / ctx["temp"] + ctx["bias"]))
    w, h = img.size
    if min(w, h) < 32:
        return {"skip": "too-small"}
    dg = degenerate_reason(img)
    if dg:
        return {"skip": "degenerate-" + dg}
    z_std = ctx["logit"](preprocess(img, S, C))
    z, used_tta = z_std, False
    if tta_cfg.get("enabled"):
        s = calibrate(z_std)
        if (tta_cfg["band_lo"] <= s <= tta_cfg["band_hi"]
                and min(w, h) >= tta_cfg.get("min_side", C)):
            z = (z_std + ctx["logit"](preprocess_native(img, C))) / 2.0
            used_tta = True
    return {"tta": used_tta, "logit_std": z_std, "logit": z, "score": calibrate(z)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="clean | web | hard")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ctx = load_ctx()

    d = os.path.join(HERE, "images", a.set)
    files = sorted(os.listdir(d))
    if a.limit:
        files = files[: a.limit]
    out, t0 = [], time.time()
    for i, f in enumerate(files):
        img = Image.open(os.path.join(d, f)).convert("RGB")
        w, h = img.size
        parts = f.split("__")
        rec = {"file": f, "label": 1 if f.startswith("ai__") else 0,
               "source": parts[1] if len(parts) > 2 else "-", "w": w, "h": h}
        rec.update(score_image(img, ctx))
        out.append(rec)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(files)}  {el:.0f}s  ({el/(i+1)*1000:.0f} ms/img)",
                  file=sys.stderr, flush=True)
    dest = a.out or os.path.join(HERE, f"scores_{a.set}.json")
    # Stamp the model into the file. A scores file that does not say which ONNX produced it
    # cannot contradict a wrong memory, and mine was wrong for a whole sweep.
    _m = json.load(open(MANIFEST))
    sio.dump(dest, out, {"version": _m["version"], "model_file": os.path.basename(MODEL),
                         "sha256": hashlib.sha256(open(MODEL, "rb").read()).hexdigest(),
                         "bias": _m.get("calibration", {}).get("bias"),
                         "temperature": _m.get("calibration", {}).get("temperature"),
                         "input_size": _m.get("input_size")})
    scored = [r for r in out if "score" in r]
    print(f"{a.set}: {len(scored)} scored, {len(out)-len(scored)} skipped -> {dest}")


if __name__ == "__main__":
    main()
