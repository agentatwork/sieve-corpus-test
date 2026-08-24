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
score.py         the extension's scoring path, in Python, validated against its own reference
analyze.py       per-generator / per-source tables at the fixed 0.65 threshold
meta_probe.mjs   how much the metadata short-circuit covers, using Sieve's own sniffMetadata()
manifest.json    the 320 files with labels and source clusters
scores_*.json    every per-image score, so any table here can be recomputed or disputed
```

Image bytes are not redistributed: the corpus includes research-licence face datasets and a
mugshot corpus. The SHA-256 manifest in
[c143-survey](https://github.com/agentatwork/c143-survey/blob/main/CORPUS_MANIFEST.json) lets you
rebuild it byte-identically from the public sources.

MIT.
