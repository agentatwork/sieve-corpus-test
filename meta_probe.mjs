// How much does Sieve's metadata short-circuit actually cover, once an image has
// been through the web?
//
// sniffMetadata() short-circuits to 0.99 confidence on a structural provenance
// marker. That is the highest-confidence verdict the extension can produce, so
// it is worth knowing (a) whether any real photograph triggers it and (b) how
// many AI images still carry a marker after the ordinary delivery path a browser
// actually sees. Uses Sieve's own exported function, unmodified.

import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { sniffMetadata } from "/tmp/sieve/extension/src/forensics.js";

const SETS = { clean: "images/clean", web: "images/web", hard: "images/hard" };
const HERE = "/home/agent/work/sieve-test";

// The originals, before I re-encoded anything — the best case for metadata.
const ORIG_ROOTS = ["/home/agent/work/aidetect/data/all",
                    "/home/agent/work/aidetect/data/ai",
                    "/home/agent/work/aidetect/data/real"];
const manifest = JSON.parse(readFileSync(join(HERE, "manifest.json"), "utf8"));

function origPath(f) {
  for (const r of ORIG_ROOTS) {
    try { readFileSync(join(r, f), { flag: "r" }); return join(r, f); } catch {}
  }
  return null;
}

const rows = [];
let origAi = 0, origReal = 0, origAiHit = 0, origRealHit = 0;
const reasons = {};
for (const m of manifest) {
  const p = origPath(m.file);
  if (!p) continue;
  const hit = sniffMetadata(new Uint8Array(readFileSync(p)));
  if (m.label === 1) { origAi++; if (hit.hit) { origAiHit++; reasons[hit.reason] = (reasons[hit.reason] || 0) + 1; } }
  else { origReal++; if (hit.hit) { origRealHit++; reasons["FP:" + hit.reason] = (reasons["FP:" + hit.reason] || 0) + 1; } }
  if (hit.hit) rows.push({ file: m.file, label: m.label, source: m.source, reason: hit.reason, set: "original" });
}

console.log("ORIGINAL FILES, as downloaded from the dataset");
console.log(`  AI   : ${origAiHit}/${origAi} carry a structural provenance marker`);
console.log(`  real : ${origRealHit}/${origReal} do  <- any non-zero here is a 0.99 false positive`);
console.log("  reasons:", JSON.stringify(reasons));

for (const [name, dir] of Object.entries(SETS)) {
  let ai = 0, real = 0, aiHit = 0, realHit = 0;
  for (const f of readdirSync(join(HERE, dir))) {
    const isAi = f.startsWith("ai__");
    const hit = sniffMetadata(new Uint8Array(readFileSync(join(HERE, dir, f))));
    if (isAi) { ai++; if (hit.hit) aiHit++; } else { real++; if (hit.hit) realHit++; }
    if (hit.hit) rows.push({ file: f, label: isAi ? 1 : 0, reason: hit.reason, set: name });
  }
  console.log(`\n${name.toUpperCase()} (my delivery pipeline)`);
  console.log(`  AI   : ${aiHit}/${ai}`);
  console.log(`  real : ${realHit}/${real}`);
}

writeFileSync(join(HERE, "meta_probe.json"), JSON.stringify(rows, null, 1));
console.log(`\n${rows.length} marker hits written to meta_probe.json`);
