#!/usr/bin/env python3
"""Package the 40-image face regression set the maintainer asked for in issue #48.

  "I'd gladly take the 40-image face set (10 FFHQ + 24 StyleGAN + conditions) as a standing
   regression set -- a link or attachment here is perfect, and it will be scored in-extension
   before every release like the other reported images."

Three of the four real-face clusters in my corpus can be shipped as bytes; one cannot, and the
StyleGAN half *must* be, for opposite reasons:

  ffhq-256      shipped.  FFHQ's authors selected only permissively-licensed Flickr photos
                          (CC-BY / CC-BY-SA / public domain / US-gov); the set is mirrored
                          publicly as bitmind/ffhq-256.
  celeb-a-hq    hashes.   Not part of the request. Mirror is cc-by-4.0 if it is ever wanted.
  lfw           hashes.   Not part of the request.
  idoc-mugshots NEVER.    Photographs of identifiable incarcerated people. Public-record
                          status is not a reason to vendor them into a browser extension's
                          release suite, and a maintainer who pulls this cluster by hash
                          should know what is in it before it lands in a public repo.
  stylegan-tpdne shipped. Fetched one at a time from thispersondoesnotexist.com, which mints
                          a fresh face per request -- so unlike every other cluster these are
                          NOT reproducible from a hash. Ship the bytes or the number can never
                          be re-run by anyone, including me. No real person is depicted.

Writes faces-regression/ with the images, per-file sha256, and the rebuild pointers for the
clusters whose bytes stay here.
"""
import hashlib, json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "faces-regression")

SHIP = {"ffhq-256": "bitmind/ffhq-256", "stylegan-tpdne": None}
HASH_ONLY = {
    "celeb-a-hq": ("bitmind/celeb-a-hq", "cc-by-4.0 on the mirror"),
    "lfw": ("bitmind/lfw", "research use; faces of public figures"),
    "idoc-mugshots-images": ("bitmind/idoc-mugshots-images",
                             "identifiable incarcerated people -- bytes deliberately withheld"),
}
# real faces live in the pooled condition dirs, StyleGAN in its own sg_* dirs
DIRS = {"clean": ("clean", "sg_clean"), "web": ("web", "sg_web"), "hard": ("hard", "sg_hard")}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    index = {"purpose": "standing face regression set for Sieve, requested in issue #48",
             "conditions": {}, "withheld": {}}

    for cond, (pooled, sg) in DIRS.items():
        dest = os.path.join(OUT, cond)
        os.makedirs(dest, exist_ok=True)
        entry = {}
        for src_dir, clusters in ((pooled, ["ffhq-256"]), (sg, ["stylegan-tpdne"])):
            d = os.path.join(HERE, "images", src_dir)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not any(c in f for c in clusters):
                    continue
                shutil.copy2(os.path.join(d, f), os.path.join(dest, f))
                entry[f] = {"sha256": sha(os.path.join(dest, f)),
                            "label": "ai" if f.startswith("ai__") else "real"}
        index["conditions"][cond] = entry
        print(f"  {cond:6} {len(entry):3} images -> faces-regression/{cond}/")

    for cluster, (mirror, why) in HASH_ONLY.items():
        files = {}
        for cond, (pooled, _) in DIRS.items():
            d = os.path.join(HERE, "images", pooled)
            for f in sorted(os.listdir(d)):
                if cluster in f:
                    files.setdefault(cond, {})[f] = sha(os.path.join(d, f))
        index["withheld"][cluster] = {"source": mirror, "reason": why, "sha256": files}
        n = sum(len(v) for v in files.values())
        print(f"  {cluster:22} {n:3} hashes only -- {why}")

    json.dump(index, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1, sort_keys=True)
    total = sum(len(v) for v in index["conditions"].values())
    print(f"\n  {total} image files + MANIFEST.json in {OUT}")


if __name__ == "__main__":
    main()
