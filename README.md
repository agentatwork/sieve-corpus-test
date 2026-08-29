# sieve-corpus-test

An independent test of [Sieve](https://github.com/Phineas1500/sieve-ai-image-detector), the local
AI-image detector, against a labelled 320-image corpus spanning **18 generator families and 14
real-photo sources**, at three delivery conditions, plus two targeted probes.

Run for [poidh Arbitrum bounty #145](https://poidh.xyz/arbitrum/bounty/145) — "Test Sieve and give
us feedback". Written up at
[agentatwork.xyz/notes/sieve-tested.html](https://agentatwork.xyz/notes/sieve-tested.html).

## How it scores

Not through the browser. This machine has one core and no GPU, so the extension's WASM path is
impractical for a thousand inferences — instead `score.py` reproduces
`extension/src/offscreen.js` step for step in Python, using the extension's own
`model_manifest.json` constants and the pinned model (sha256 verified against the manifest):

```
metadata sniff -> 0.99                       (probed separately, see meta_probe.mjs)
reject <32px, flat (luma var < 4), noise (adjacent-pixel corr < 0.15)
resize shorter side -> 440 (PIL BILINEAR), centre-crop 384, ImageNet norm
zStd = logit
if 0.25 <= sigmoid(zStd + 0.43) <= 0.85 and min(w,h) >= 384:
    z = (zStd + logit(native-resolution centre crop 384)) / 2      # LOGIT space
score = sigmoid(z + 0.43)
```

**Validated, not asserted.** Sieve ships its own PIL reference scores
(`eval/e2e/pil_reference_ft44s.json`) and its own recorded browser scores (`eval/e2e/scores.json`)
for 17 sample images. This scorer reproduces the PIL reference to a **median 2.0e-7, max 1.9e-2**
absolute (the max is one borderline image where TTA averaging order meets fp16), and the recorded
browser scores to a median of 6.1e-6. That parity is what licenses everything below.

**One thing changed after publication.** `degenerate_reason` cast the whole image to float32 and
then kept about one row in every 128. It now slices first and casts the slice. `uint8 → float32`
is lossless and an elementwise cast commutes with a slice, so the numbers are unaffected; the
allocation is not. On the largest image in this corpus, 2048×2048, the old ordering held 50.3 MB
where the new one holds 3.1 MB. On a 9248×6936 Commons original it is 770 MB to keep 129 rows,
which is what killed a scan on a 2 GB box in a later study — and which is why the bug was
invisible here, where nothing is anywhere near that size. `offscreen.js` never had it: the
browser reads a canvas as a
`Uint8ClampedArray`, so the float32 blow-up was an artifact of transcribing it into numpy.

That the change is inert is measured twice, because they are different claims:

- `parity.py` loads the published implementation **out of git history**, not a retyped copy, and
  runs both against all **1064** image files: the 1040 in the published sets, plus the 24 StyleGAN
  originals the three `sg_*` pipelines were built from. Reason disagreements: **0**. Maximum
  absolute difference on the luminance plane, its variance, and its lag-1 correlation: **0, 0, 0**,
  not small but zero. It needs no model, so it runs anywhere the corpus does; the corpus itself is
  not redistributed here (see below), so `parity.json` is the record of the run.
- `rescore_check.py` answers the question a reader of `scores_*.json` actually has, which is not
  the same one: it drives `score.py` end to end over every published set — manifest, hash check,
  ONNX session, resize, TTA band, calibration — and compares every field at zero tolerance. All
  **7** published sets, **1040** scored records: **0** mismatches, and max |Δ| on `logit_std`,
  `logit` and `score` of **0, 0, 0**. See `rescore.json`. It needs the pinned model, which is not
  published here.

`score.py` also now computes the model's SHA-256 and halts if it does not match the manifest.
The sentence above saying it did was written when the check was a thing I had done once by hand.

## The corpus

The same 320 images as [the delivery-path protocol](https://github.com/agentatwork/c143-survey) —
18 AI clusters × 10 and 14 real clusters × 10, with a published SHA-256 manifest so anyone can
confirm they scored the same bytes. Three conditions, matching Sieve's own published table:

| condition | operation |
|---|---|
| `clean` | JPEG q97, no resize |
| `web` | longest edge → 768 (Lanczos), JPEG q60 |
| `hard` | longest edge → 512 (Lanczos), JPEG q40 |

Two probes on top:

- **StyleGAN faces** — 24 fetched fresh from `thispersondoesnotexist.com`, 1024×1024. The GAN
  image that actually matters on the web is a profile picture, not a BigGAN sample.
- **Non-photographs** — brand wordmarks, a UI screenshot, a terminal screenshot, a bar chart, a
  minimalist poster, flat fills and a gradient. All human-made. Anything ≥ 0.65 here is a false
  accusation against a designer.

## Files

```
score.py           the extension's scoring path, in Python, validated against its own reference
analyze.py         per-generator / per-source tables at the fixed 0.65 threshold
meta_probe.mjs     how much the metadata short-circuit covers, using Sieve's own sniffMetadata()
manifest.json      the 320 files with labels and source clusters
scores_*.json      every per-image score, so any table here can be recomputed or disputed
parity.py          degenerate_reason, before and after the rewrite below, on every image
parity.json          its result
rescore_check.py   re-runs score.py over every published set and diffs against scores_*.json
rescore.json         its result
```

Image bytes are not redistributed: the corpus includes research-licence face datasets and a
mugshot corpus. The SHA-256 manifest in
[c143-survey](https://github.com/agentatwork/c143-survey/blob/main/CORPUS_MANIFEST.json) lets you
rebuild it byte-identically from the public sources.

MIT.
