#!/usr/bin/env python3
"""Compute both TTA views' logits once per image, so every gating variant is arithmetic.

The shipped scorer decides *whether* to take a second view and then throws the second
logit away when it declines. To compare gating rules you only need the two logits, and
they do not depend on the rule -- so this runs the model twice per image, stores both,
and `variants.py` derives baseline / A / B from the same numbers. 1920 inferences instead
of 5760, and every variant is compared on bit-identical model output.

  z_std  resize shorter side -> 440 (BILINEAR), centre-crop 384      [shipped]
  z_nat  min(w,h) >= 384: 1:1 centre-crop 384                        [shipped native view]
         min(w,h) <  384: resize shorter side -> 384, centre-crop 384 [the variant's definition]

Writes dual_<set>.json. Resumable: an existing file is loaded and only missing files
are scored, because this box is one core and a 30-minute sweep should not have to
start over.
"""
import json, os, sys, time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import HERE, load_ctx, preprocess, preprocess_native, degenerate_reason


def dual_logits(img, ctx):
    S, C = ctx["S"], ctx["C"]
    w, h = img.size
    if min(w, h) < 32:
        return {"skip": "too-small"}
    dg = degenerate_reason(img)
    if dg:
        return {"skip": "degenerate-" + dg}
    z_std = ctx["logit"](preprocess(img, S, C))
    if min(w, h) >= C:
        z_nat = ctx["logit"](preprocess_native(img, C))   # shipped native view, unchanged
    else:
        z_nat = ctx["logit"](preprocess(img, C, C))       # minimum upscale, full frame
    return {"w": w, "h": h, "z_std": z_std, "z_nat": z_nat}


def run(setname, ctx):
    d = os.path.join(HERE, "images", setname)
    dest = os.path.join(HERE, f"dual_{setname}.json")
    done = {}
    if os.path.exists(dest):
        done = {r["file"]: r for r in json.load(open(dest))}
    files = sorted(os.listdir(d))
    todo = [f for f in files if f not in done]
    print(f"{setname}: {len(files)} files, {len(done)} cached, {len(todo)} to score", flush=True)
    t0 = time.time()
    for i, f in enumerate(todo):
        img = Image.open(os.path.join(d, f)).convert("RGB")
        parts = f.split("__")
        rec = {"file": f, "label": 1 if f.startswith("ai__") else 0,
               "source": parts[1] if len(parts) > 2 else "-"}
        rec.update(dual_logits(img, ctx))
        done[f] = rec
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(todo)}  {el:.0f}s  ({el/(i+1)*1000:.0f} ms/img)", flush=True)
            json.dump([done[k] for k in sorted(done)], open(dest, "w"))
    json.dump([done[k] for k in sorted(done)], open(dest, "w"))
    print(f"{setname}: -> {dest}", flush=True)


if __name__ == "__main__":
    ctx = load_ctx()
    for s in (sys.argv[1:] or ["hard", "web", "clean", "sg_hard", "sg_web", "sg_clean", "graphics"]):
        run(s, ctx)
