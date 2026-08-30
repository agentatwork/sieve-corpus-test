# Pre-registration — does opening the TTA gate fix Sieve's face errors?

Written **before** the variant was run. Committed before the results file exists, so the
decision rule cannot be chosen after seeing the numbers.

## Background

`extension/src/offscreen.js` fires its second (test-time-augmentation) view only when

```js
band_lo <= sigmoid(z_std + bias) <= band_hi     // 0.25 .. 0.85
&& min(w, h) >= min_side                        // 384
```

My corpus test found Sieve's worst behaviour on **face crops at `hard`** (longest edge 512,
JPEG q40): 4 of 10 FFHQ real faces flagged AI, at 0.997 / 0.942 / 0.788 / 0.705, while 7 of 24
StyleGAN fakes got through. I listed the `min_side` gate as a contributing mechanism, on the
grounds that the thumbnail regime is exactly where uncertainty repair is switched off.

## The bound, stated first

Counting the images the gate can *possibly* touch, from the shipped baseline scores:

| condition | corpus | sub-384px | TTA fired | **blocked by `min_side` alone** |
|---|---|---|---|---|
| clean | 320 | 153 | 7 | 19 |
| web | 320 | 154 | 10 | 23 |
| hard | 320 | 187 | 14 | **29** |

So at `hard` the maximum reachable set is **29 images, 9.1% of the corpus** — 15 AI, 14 real,
of which **6 are real faces**.

**And the two worst face false positives are not in that set.** The FFHQ images scoring 0.997
and 0.942 are outside the 0.25–0.85 band, so no `min_side` change can reach them. Of the four
face false positives, this fix can touch **two** — 0.788 and 0.705 — and they are the two least
severe. I am recording this now because it materially weakens a claim I already published, and
I would rather correct it than let the experiment quietly re-frame it.

Per-image leverage on pooled balanced accuracy at `hard`: one AI image = 0.278 pt, one real
image = 0.357 pt. Nothing here can move BA by more than a few points in either direction.

## Variants

All three share the same two logits per image, computed once:

- `z_std` — shipped path: resize shorter side to 440 (BILINEAR), centre-crop 384.
- `z_nat` — shipped native path when `min(w,h) >= 384`: 1:1 centre crop 384.
  When `min(w,h) < 384` the 1:1 crop does not exist, so the second view is defined as
  **resize shorter side to 384, centre-crop 384** — the minimum upscale, full-frame. This is
  still a genuinely different view from `z_std` (less upscaling, wider framing) and it needs no
  padding, which the model has never seen. For `min(w,h) >= 384` it is byte-identical to shipped
  behaviour, so the proposal changes nothing for large images.

| variant | rule |
|---|---|
| **baseline** | shipped: band AND `min_side >= 384` |
| **A** | band only — `min_side` dropped. The minimal change. |
| **B** | no gate at all — always average both views. The ceiling of what a second view can do. |

B is not a proposal; it is a measurement of headroom. If B cannot fix the confident face false
positives, then no gating change can and the remedy has to be training data or calibration.

## Decision rule

I will propose variant A upstream **only if all four hold**:

- **P1** pooled balanced accuracy at `hard` does not decrease (Δ ≥ 0.0 pt)
- **P2** real-face specificity at `hard` (ffhq-256 + celeb-a-hq + lfw + idoc-mugshots, n = 40)
  improves by at least one image (≥ +2.5 pt)
- **P3** StyleGAN recall at `hard` (n = 24) loses no more than one image
- **P4** pooled BA at `clean` and at `web` each decrease by no more than 0.5 pt

If any fail, I publish the negative result and do not propose the change. A negative is still
worth reporting: "the obvious fix does not work, here is the measurement" saves the maintainer
the afternoon I would have spent.

## What would make me distrust my own result

- If variant A changes the score of any image with `min(w,h) >= 384`. That is a bug, not a
  finding — the variant must reduce to shipped behaviour there. Asserted in the harness.
- If the recomputed baseline does not reproduce `scores_*.json` exactly. Also asserted.
