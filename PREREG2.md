# Pre-registration 2 — is the second view worth more as a doubt signal than as a vote?

Written **before** the ft58s numbers exist. The v0.12.0 sweep is running as I type this; no
result below has been seen.

## Why there is a second pre-registration

PREREG.md asked whether opening the TTA gate helps. On ft44s the answer was no: variant A
failed its own P4, and the threshold-free view showed why — A moves discrimination by
|ΔAUC| ≤ 0.002, so its +2.74 balanced accuracy at `hard` was threshold jitter, not skill.
Re-fitting the threshold did not rescue it: of six held-out transfers, five were negative.

Two things then changed the question:

1. **I was two models stale.** Upstream shipped v0.11.0 (ft5s) and v0.12.0 (ft58s) while I was
   measuring ft44s, and moved the calibration bias 0.43 → 0.30. Every number in PREREG.md is
   about a model Sieve no longer ships. The sweep is being redone on ft58s.
2. **The maintainer has variant A queued.** Issue #53 lists, for v0.13, "run the TTA second
   view on an upscaled native crop instead of skipping it below 384px" — the change PREREG.md
   measured as a regression. That makes a negative result actionable rather than academic,
   *provided* it is measured on the model that will ship it.

And in #42 the maintainer named, explicitly, the evidence that would change their mind:

> If you have evidence the multi-view ensemble disagrees *within itself* on real web images
> (rather than all views flipping together), that would change my mind.

Nobody has answered that. I have both views' logits for every image at every degradation
level, so I can — and the answer decides between two different designs, not two thresholds.

## The reframe

Averaging two views is a *vote*: it assumes both views are noisy estimates of one truth, so
the mean is better than either. Everything measured so far says that assumption is close to
worthless here — the mean ranks no better than the single view.

But a second view can be used a completely different way: as a *doubt signal*. If the two
views disagree, the model is standing on something unstable, and the right product behaviour
is to say "unsure" rather than to average two shaky numbers into one confident-looking one.
That is the design the maintainer said they preferred in #42 — "widens the band or forces
'unsure' only on inputs that look already-degraded" — and it costs the same two inferences
that averaging costs, so the accuracy question is the only thing separating them.

## The measurement

For every image, with both views calibrated through the shipped sigmoid:

```
d = | cal(z_std) - cal(z_nat) |          disagreement between the two views
e = 1 if the shipped baseline verdict at 0.65 is wrong, else 0
```

**Primary:** AUC of `d` as a predictor of `e`, per condition.

**The control that decides whether any of this is worth paying for.** Sieve already has a free
uncertainty signal: distance from the threshold, `u = |cal(z_std) - 0.65|`, which costs zero
extra inference because it needs only the view already computed. If `u` predicts errors as
well as `d` does, then the second view earns nothing as a doubt signal either, and the honest
recommendation is to spend the 2× inference budget somewhere else. **I will report `u` beside
`d` every time `d` appears.** A disagreement signal that cannot beat "the score was near the
threshold" is not a finding, it is a more expensive way to measure the same thing.

## Decision rule

I will propose the doubt-signal design upstream **only if both hold**:

- **Q1** AUC(`d` → error) ≥ 0.70 at `hard`, and ≥ 0.65 at `clean` and `web`.
- **Q2** `d` beats the free control: AUC(`d`) − AUC(`u`) ≥ +0.03 at **every** one of the three
  conditions. Not on average — averaging is how a win on one condition hides a loss on two.

Secondary, reported either way, not a gate: at a 5% abstention rate (flag the 5% of images
with the largest `d`), what share of all errors is captured, and what is the lift over the 5%
a random abstention would catch?

If Q1 fails, the second view carries no usable doubt signal and I report that. If Q1 passes
but Q2 fails, the *right* recommendation is an abstention band on the existing score — which
is free, needs no second inference, and I will say so plainly even though it makes the second
view, and both of my pre-registrations, irrelevant to the fix.

## What would make me distrust this

- If `d` is near-zero for almost every image, the AUC is being computed over ties and means
  little. I will report the share of images with `d < 0.01` alongside it.
- `e` is defined against the *shipped* baseline verdict, so `d` and `e` are not independent —
  `d` is built from `z_std`, which also determines `e`. This is exactly why Q2 exists: the
  control `u` is built from `z_std` alone, so anything `d` wins over `u` is attributable to
  `z_nat` and nothing else.
- Small n. Errors at `clean` are rare (~9% of 320 ≈ 29 images), so the clean AUC will have a
  wide interval. I will report the error count per condition next to each AUC and will not
  read a 0.02 difference on 29 events as real.
