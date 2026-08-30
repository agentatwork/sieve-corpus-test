#!/usr/bin/env python3
"""Re-run score.py over a published set and require the published numbers back, exactly.

parity.py answers "did degenerate_reason change what it returns". This answers the question a
reader of scores_clean.json actually has: *are the published scores still what this code
produces*. The two are not the same claim. parity.py compares a three-way branch and needs no
model; this drives the real entry point end to end -- manifest, sha256 check, ONNX session,
resize, TTA band, calibration -- and compares every float in the published file.

It shells out to score.py rather than importing it, so the thing under test is the command in
README.md and not a hand-assembled approximation of it.

The sets are discovered from the published files rather than listed here. A hard-coded list is
a second copy of "which results were published", and the copy that gets forgotten is the one
that would have caught a set going stale. [[one-rule-two-copies]]

Needs the pinned model, which is not published here (43 MB); see README.md. About ten minutes
for all seven sets on one core -- long enough that the first attempt at this was killed partway
through and left nothing behind. So each set is written to the output file as it finishes, and
a re-run skips sets already recorded. The cache is keyed on the SHA-256 of score.py: a resume
that reused a result produced by a different scorer would be worse than no resume at all, since
the file would read as a full pass. [[long-scans-need-resume]]

  python3 rescore_check.py                 # every published set, resuming if interrupted
  python3 rescore_check.py --sets clean
  python3 rescore_check.py --force         # ignore anything already recorded
"""
import argparse, glob, hashlib, json, os, subprocess, sys, tempfile
import sio

HERE = os.path.dirname(os.path.abspath(__file__))
SCORER = os.path.join(HERE, "score.py")
# Compared per record. Floats are compared exactly: this is the same model, the same weights and
# the same arithmetic, so any tolerance at all would be a place for a real change to hide.
FLOAT_FIELDS = ("logit_std", "logit", "score")
EXACT_FIELDS = ("label", "source", "w", "h", "skip", "tta")


def check_set(name, scorer_sha):
    """Re-score one published set and return a record of how it compared."""
    published = os.path.join(HERE, f"scores_{name}.json")
    fresh = tempfile.mktemp(suffix=f"_{name}.json")
    r = subprocess.run([sys.executable, SCORER, "--set", name, "--out", fresh], cwd=HERE)
    if r.returncode:
        sys.exit(f"score.py --set {name} exited {r.returncode}")

    pub = {x["file"]: x for x in sio.records(published)}
    new = {x["file"]: x for x in sio.records(fresh)}
    os.unlink(fresh)

    missing = sorted(set(pub) - set(new))
    added = sorted(set(new) - set(pub))
    worst, mismatches = {k: 0.0 for k in FLOAT_FIELDS}, []
    for f in sorted(set(pub) & set(new)):
        p, n = pub[f], new[f]
        for k in EXACT_FIELDS:
            if p.get(k) != n.get(k):
                mismatches.append(dict(file=f, field=k, published=p.get(k), rescored=n.get(k)))
        for k in FLOAT_FIELDS:
            if k in p or k in n:
                # A field present in one and absent in the other is a mismatch, not a delta of
                # zero -- .get(k, 0.0) would have quietly called that agreement.
                if k not in p or k not in n:
                    mismatches.append(dict(file=f, field=k,
                                           published=p.get(k), rescored=n.get(k)))
                else:
                    worst[k] = max(worst[k], abs(p[k] - n[k]))

    return dict(set=name, scorer_sha256=scorer_sha,
                n_published=len(pub), n_rescored=len(new),
                missing=missing, added=added, mismatches=mismatches,
                max_abs_delta=worst)


def set_ok(r):
    return (not (r["missing"] or r["added"] or r["mismatches"])
            and max(r["max_abs_delta"].values()) == 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="", help="comma-separated; default: every published set")
    ap.add_argument("--out", default=os.path.join(HERE, "rescore.json"))
    ap.add_argument("--force", action="store_true", help="re-run sets already recorded")
    a = ap.parse_args()

    all_sets = sorted(os.path.basename(p)[len("scores_"):-len(".json")]
                      for p in glob.glob(os.path.join(HERE, "scores_*.json")))
    if not all_sets:
        sys.exit("no scores_*.json found")
    if a.sets:
        sets = [s.strip() for s in a.sets.split(",") if s.strip()]
        for s in sets:
            if s not in all_sets:
                sys.exit(f"no published scores to compare against: scores_{s}.json")
    else:
        sets = all_sets

    scorer_sha = hashlib.sha256(open(SCORER, "rb").read()).hexdigest()
    done = {}
    if os.path.exists(a.out) and not a.force:
        try:
            prev = json.load(open(a.out))
        except json.JSONDecodeError:      # killed mid-write; start over rather than guess
            prev = {}
        done = {r["set"]: r for r in prev.get("results", [])
                if r.get("scorer_sha256") == scorer_sha}

    def flush(per):
        """Write after every set, so an interrupted run leaves the sets it did finish.

        `passed` means every *published* set reproduced -- not every set this invocation
        happened to ask for. `--sets clean` verifies one seventh of the results and must not
        leave behind a file that reads as a full pass; I misread exactly that file twice while
        building this. So `complete` is measured against the glob, and a narrow run writes
        passed=false with `checked` naming what it did cover. [[completeness-check-granularity]]
        """
        checked = sorted(r["set"] for r in per)
        complete = checked == all_sets
        ok = complete and all(set_ok(r) for r in per)
        json.dump(dict(passed=ok, tolerance=0.0, complete=complete,
                       published_sets=all_sets, checked=checked,
                       scorer_sha256=scorer_sha,
                       n_images=sum(r["n_published"] for r in per),
                       results=per), open(a.out, "w"), indent=1)
        return ok

    per, ok = [], False       # `--sets ,` yields no sets; exit 1 rather than NameError
    for s in sets:
        if s in done:
            print(f"  {s:<10} already recorded against this score.py -- skipping")
            per.append(done[s])
        else:
            per.append(check_set(s, scorer_sha))
        ok = flush(per)

    for r in per:
        print(f"  {r['set']:<10} {r['n_published']:>4} published  "
              f"mismatches {len(r['mismatches'])}  "
              f"max |delta| " + " ".join(f"{k} {v:.17g}"
                                         for k, v in r["max_abs_delta"].items()))
    print(f"{sum(r['n_published'] for r in per)} images across {len(per)} of "
          f"{len(all_sets)} published sets  "
          + ("PASS" if ok else "FAIL") + f"  -> {os.path.relpath(a.out, HERE)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
