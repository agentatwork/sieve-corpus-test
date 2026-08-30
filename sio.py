"""Read and write score files so that every number carries the model that produced it.

The scores_*.json files used to be a bare JSON list of per-image records with nothing saying
which ONNX had produced them. That is how I spent a full sweep measuring ft44s (v0.10.0) while
upstream had already shipped ft5s and then ft58s, moving the calibration bias 0.43 -> 0.30 on
the way: the files on disk looked exactly the same before and after, so nothing could contradict
me. A version I have to remember is a version I will get wrong.

So the on-disk format is now:

    {"model": {"version", "model_file", "sha256", "bias", "temperature", "input_size"},
     "written_utc": "...",
     "scores": [ ...per-image records... ]}

`load` accepts both shapes and returns (meta, records) either way, with meta = {} for a legacy
bare list -- old files stay readable, they just cannot claim a provenance they never recorded.
Callers that want to *rely* on the version should use `require_version`, which fails loudly
rather than silently comparing across model generations.

One loader, one writer, six call sites. Hand-copying this into each reader is how the two
copies drift apart and the divergence lands on whichever branch the check does not reach.
"""
import json, os, time


def load(path):
    """(meta, records). meta is {} for a legacy bare-list file."""
    d = json.load(open(path))
    if isinstance(d, list):
        return {}, d
    return d.get("model", {}), d["scores"]


def records(path):
    """Just the records, for callers that genuinely do not care about provenance."""
    return load(path)[1]


def dump(path, records, meta):
    json.dump({"model": meta,
               "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "scores": records}, open(path, "w"))


def require_version(paths, want=None):
    """Assert every file was produced by the same model, and return that version.

    Raises on a legacy file with no stamp rather than assuming it matches: 'no version
    recorded' and 'the version I expected' are different states, and treating the first as
    the second is the whole failure this module exists to prevent.
    """
    seen = {}
    for p in paths:
        meta, _ = load(p)
        v = meta.get("version")
        if v is None:
            raise SystemExit(
                f"{os.path.basename(p)} records no model version (pre-{__name__} file).\n"
                f"Regenerate it with score.py rather than assuming it matches the others.")
        seen.setdefault(v, []).append(os.path.basename(p))
    if len(seen) > 1:
        lines = "\n".join(f"  {v}: {', '.join(f)}" for v, f in sorted(seen.items()))
        raise SystemExit(f"score files span {len(seen)} model versions:\n{lines}")
    got = next(iter(seen))
    if want and got != want:
        raise SystemExit(f"score files are {got}, but the model on disk is {want}. Regenerate.")
    return got
