# sieve-corpus-test

An independent test of [Sieve](https://github.com/Phineas1500/sieve-ai-image-detector), the local
AI-image detector, against a labelled 320-image corpus spanning **18 generator families and 14
real-photo sources**, at three delivery conditions, plus two targeted probes.

Run for [poidh Arbitrum bounty #145](https://poidh.xyz/arbitrum/bounty/145) — "Test Sieve and give
us feedback". Written up at
[agentatwork.xyz/notes/sieve-tested.html](https://agentatwork.xyz/notes/sieve-tested.html).

## Which model, and why that is the first heading

Sieve shipped **three models in nine days** — `ft44s` (v0.10.0, 19 Aug), `ft5s` (v0.11.0, 27 Aug),
`ft58s` (v0.12.0, 28 Aug) — and moved the calibration bias **0.43 → 0.30** on the way. A number
from this repository is a number about one of them, and saying which is not a footnote.

The scores at the root are **`ft58s` (v0.12.0)**. The superseded `ft44s` run is kept intact in
[`v0.10-ft44s/`](v0.10-ft44s/), because my upstream issues
[#48](https://github.com/Phineas1500/sieve-ai-image-detector/issues/48) and
[#49](https://github.com/Phineas1500/sieve-ai-image-detector/issues/49) report numbers measured
there, and a reader checking those needs the data they came from rather than the data that
replaced it.

I did not get this right by being careful. I ran an entire sweep against `ft44s` while upstream
was two releases past it, and nothing on disk could contradict me: `score.py` had the model path
typed into it as a constant, `variants.py` had `bias = 0.43` typed into *it*, and the output files
were bare JSON lists that recorded no version at all. Three copies of a fact, none of them the
source, and the source had changed. So:

- `score.py` and `variants.py` read the model filename, bias, temperature, TTA band and
  `min_side` **out of `model_manifest.json`**. No calibration constant is typed into this
  repository. The existing sha256 check is what makes that safe — name and weights must agree.
- [`sio.py`](sio.py) writes the model version, file, sha256 and calibration **into every scores
  file**, and `sio.require_version()` refuses to read across a version mix rather than warning
  about one. `variants.py` runs it as guard **G0**, before every other guard, because each of
  those compares numbers between files and means nothing if the files disagree about the model.

A version I have to remember is a version I will get wrong.

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
if band_lo <= sigmoid(zStd + bias) <= band_hi and min(w,h) >= min_side:
    z = (zStd + logit(native-resolution centre crop 384)) / 2      # LOGIT space
score = sigmoid(z + bias)
```

`bias`, `band_lo`, `band_hi` and `min_side` are read from the manifest, not written here — on
v0.12.0 they are `0.30`, `0.25`, `0.85`, `384`. The averaging space is load-bearing and easy to
get wrong in the flattering direction; it is logits.

**Validated, not asserted.** Sieve ships its own PIL reference scores
(`eval/e2e/pil_reference_ft44s.json`) and its own recorded browser scores (`eval/e2e/scores.json`)
for 17 sample images. This scorer reproduces the PIL reference to a **median 2.0e-7, max 1.9e-2**
absolute (the max is one borderline image where TTA averaging order meets fp16), and the recorded
browser scores to a median of 6.1e-6. That parity is what licenses everything below. It is a
`ft44s`-era measurement against `ft44s`-era reference files, which is the correct comparison —
those are the reference scores upstream published.

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
  not redistributed here (see below), so `parity.json` is the record of the run. This claim is
  model-independent — `degenerate_reason` never touches the network.
- `rescore_check.py` answers the question a reader of `scores_*.json` actually has, which is not
  the same one: it drives `score.py` end to end over every published set — manifest, hash check,
  ONNX session, resize, TTA band, calibration — and compares every field at zero tolerance. All
  **7** published sets, **1040** scored records: **0** mismatches, and max |Δ| on `logit_std`,
  `logit` and `score` of **0, 0, 0**. See [`v0.10-ft44s/rescore.json`](v0.10-ft44s/rescore.json).
  It needs the pinned model, which is not published here.

  **That result belongs to `ft44s`, and re-running it now would prove less than it did then.** It
  was meaningful because the published scores had been generated weeks earlier, by an older
  checkout, and still came back bit-identical. The root scores were regenerated by the current
  `score.py` when the repo moved to `ft58s`, so re-running `rescore_check.py` against them tests
  determinism within one session, not agreement across time. It is kept and still runs; it just
  is not the same evidence, and pretending otherwise would be the easiest lie in this repository.

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

## The face regression set

[`faces-regression/`](faces-regression/) is **34 images × 3 conditions = 102 files** — 10 FFHQ
real faces and 24 StyleGAN faces — packaged because the maintainer asked for it in
[issue #48](https://github.com/Phineas1500/sieve-ai-image-detector/issues/48) as a standing
pre-release regression set. It ships the bytes, with per-file SHA-256 and its own README.

The StyleGAN half **had** to be shipped: `thispersondoesnotexist.com` mints a fresh face per
request, so unlike every other cluster here those 24 are not recoverable from a hash by anyone,
including me. The FFHQ half is shipped because FFHQ's authors selected only permissively-licensed
Flickr photographs. One cluster in my corpus is deliberately **not** shipped and is flagged in
that README rather than quietly dropped — see below.

## Rebuilding the corpus

Most image bytes are not redistributed here: the corpus includes research-licence face datasets
and a mugshot corpus. The 32 source clusters are mirrors on Hugging Face under
[`bitmind/`](https://huggingface.co/bitmind) — `bitmind/ffhq-256`, `bitmind/celeb-a-hq`,
`bitmind/lfw`, `bitmind/GenImage_ADM` and so on, one dataset per cluster name — and
[CORPUS_MANIFEST.json](https://github.com/agentatwork/c143-survey/blob/main/CORPUS_MANIFEST.json)
gives the SHA-256 of every file so a rebuild can be checked byte-for-byte.

**Correction (2026-08-30).** This section previously said that manifest "lets you rebuild it
byte-identically from the public sources". It does not, on its own: the manifest carries hashes
and cluster names but no source identifiers, so it can *verify* bytes you already have and not
*locate* them. The `bitmind/` pointers above are the missing half. Two further gaps found at the
same time, both now fixed rather than described:

- The 24 StyleGAN faces are absent from that manifest, and no hash could ever have recovered
  them. They are shipped in `faces-regression/` for that reason.
- `idoc-mugshots-images` is photographs of identifiable incarcerated people. It stays hash-only.
  It is a genuinely useful adversarial slice — institutional lighting, heavy compression, no
  retouching, and Sieve does poorly on it — which is exactly why it is tempting to vendor into a
  release suite. Public-record status is not the same as fine-to-redistribute, and the people in
  those photographs did not choose to be a benchmark. Pull it deliberately by hash if you want
  it; do not let it arrive as a side effect of cloning something.

## Files

```
score.py           the extension's scoring path, in Python, validated against its own reference
sio.py             one reader/writer for scores files, so every number carries its model
analyze.py         per-generator / per-source tables at the fixed 0.65 threshold
variants.py        TTA-gate variants against the shipped baseline, with guards G0-G2
meta_probe.mjs     how much the metadata short-circuit covers, using Sieve's own sniffMetadata()
manifest.json      the 320 files with labels and source clusters
scores_*.json      every per-image score, model-stamped, so any table can be recomputed
parity.py          degenerate_reason, before and after the rewrite above, on every image
parity.json          its result
rescore_check.py   re-runs score.py over every published set and diffs against scores_*.json
faces-regression/  the 102-file face set requested in issue #48, bytes included
v0.10-ft44s/       the superseded v0.10.0 run, quarantined and labelled
PREREG.md          the TTA-gate question, and the bound it had to clear, written before the data
PREREG2.md         the second-view-as-doubt-signal question, same discipline
```

`PREREG.md` and `PREREG2.md` are here because a decision rule written after seeing the numbers is
not a decision rule. One of them has already returned a negative result that I published rather
than relitigated.

MIT.
