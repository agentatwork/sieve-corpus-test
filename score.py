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
import argparse, json, os, sys, time

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "/tmp/sieve/dev_model/ft44s_best_fp16.onnx"
MANIFEST = "/tmp/sieve/extension/model_manifest.json"
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
    """offscreen.js degenerateReason(), same sampling stride and thresholds."""
    a = np.asarray(img, dtype=np.float32)
    h, w, _ = a.shape
    sy = max(1, h >> 7)
    rows = a[::sy]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="clean | web | hard")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    meta = json.load(open(MANIFEST))
    S, C = meta["resize_shorter_side"], meta["input_size"]
    cal = meta["calibration"]
    bias, temp = cal.get("bias", 0.0), cal.get("temperature", 1.0) or 1.0
    tta_cfg = meta.get("tta", {})
    sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
    calibrate = lambda z: float(sigmoid(z / temp + bias))

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1        # one core; more threads only thrash
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    logit = lambda t: float(sess.run(None, {iname: t})[0].ravel()[0])

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
        if min(w, h) < 32:
            rec["skip"] = "too-small"
        else:
            dg = degenerate_reason(img)
            if dg:
                rec["skip"] = "degenerate-" + dg
            else:
                z_std = logit(preprocess(img, S, C))
                z = z_std
                rec["tta"] = False
                if tta_cfg.get("enabled"):
                    s = calibrate(z_std)
                    if (tta_cfg["band_lo"] <= s <= tta_cfg["band_hi"]
                            and min(w, h) >= tta_cfg.get("min_side", C)):
                        z = (z_std + logit(preprocess_native(img, C))) / 2.0
                        rec["tta"] = True
                rec["logit_std"] = z_std
                rec["logit"] = z
                rec["score"] = calibrate(z)
        out.append(rec)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(files)}  {el:.0f}s  ({el/(i+1)*1000:.0f} ms/img)",
                  file=sys.stderr, flush=True)
    dest = a.out or os.path.join(HERE, f"scores_{a.set}.json")
    json.dump(out, open(dest, "w"))
    scored = [r for r in out if "score" in r]
    print(f"{a.set}: {len(scored)} scored, {len(out)-len(scored)} skipped -> {dest}")


if __name__ == "__main__":
    main()
