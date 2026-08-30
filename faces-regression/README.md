# Face regression set for Sieve

Requested by @Phineas1500 in
[issue #48](https://github.com/Phineas1500/sieve-ai-image-detector/issues/48):

> I'd gladly take the 40-image face set (10 FFHQ + 24 StyleGAN + conditions) as a standing
> regression set — a link or attachment here is perfect, and it will be scored in-extension
> before every release like the other reported images.

Here it is: **34 images × 3 delivery conditions = 102 files.** 10 FFHQ real faces (label
`real__`) and 24 StyleGAN faces from thispersondoesnotexist.com (label `ai__`). Filenames carry
the label and cluster, so a scorer can read ground truth off the path with no join.

```
clean/   JPEG q97, no resize
web/     longest edge -> 768 (Lanczos), JPEG q60
hard/    longest edge -> 512 (Lanczos), JPEG q40
```

The FFHQ images are 256×256 at every condition — a 256px image is already under 512, so `hard`
recompresses without resizing. That is the point of the set: it is the avatar/thumbnail regime,
where a 256px real face and a downscaled 1024px StyleGAN face arrive at the detector looking
much more alike than they do at full size.

`MANIFEST.json` gives the SHA-256 of every shipped file, plus hashes (no bytes) for the three
face clusters that are not shipped.

## Why these bytes are here and the other clusters' are not

The 24 StyleGAN faces **had** to be shipped. Every other cluster in the corpus is a public
dataset that a hash can verify, but these were fetched one at a time from
`thispersondoesnotexist.com`, which generates a fresh face per request. There is no dataset to
re-pull them from — if the bytes are not here, that measurement can never be reproduced by
anyone, including me. No real person is depicted.

The 10 FFHQ reals are shipped because FFHQ's authors selected only permissively-licensed Flickr
photographs (CC-BY, CC-BY-SA, public domain, US-gov), and the set is publicly mirrored as
`bitmind/ffhq-256`.

Three clusters are hash-only in `MANIFEST.json`:

| cluster | mirror | why not shipped |
|---|---|---|
| `celeb-a-hq` | `bitmind/celeb-a-hq` | not part of the request; mirror is cc-by-4.0 if wanted |
| `lfw` | `bitmind/lfw` | not part of the request |
| `idoc-mugshots-images` | `bitmind/idoc-mugshots-images` | **photographs of identifiable incarcerated people** |

**A flag on that last one, since this set is headed for a public release suite.** My corpus
draws real faces from four clusters, and one of them is an Illinois DOC mugshot corpus. It is a
good adversarial slice — institutional lighting, heavy compression, no aesthetic retouching, and
Sieve does badly on it — which is exactly why it is tempting to vendor in. Public-record status
is not the same as "fine to redistribute in a browser extension's repo," and the people in those
photographs did not choose to be a benchmark. Pull it deliberately by hash if you want it; do not
let it arrive as a side effect of cloning something.

## Scoring it

The set is scored by the same path the extension uses. From the parent directory:

```
python3 score.py --set hard        # or clean / web
```

`score.py` reads `model/model_manifest.json` and loads whichever ONNX the manifest names,
verifying its SHA-256 — so it follows upstream's model pin rather than a filename typed into the
source. That check exists because I spent a full sweep measuring `ft44s` (v0.10.0) while
upstream had shipped `ft5s` and then `ft58s`, and moved the calibration bias 0.43 → 0.30 on the
way. Any number about this set is a number about one model version; the version is printed with
the table.
