"""Compare strata-glm4ref's --dump-full-logits with a llama.cpp /completion response.

Usage:
  python compare_logits.py REF.bin RESP.json [VOCAB]

REF.bin is [step][vocab] float32 (one row per generated position).
RESP.json is llama-server's /completion body with logprobs=true.
Prints, per step, the ref's log-probabilities for llama.cpp's top-n ids, the llama
log-probabilities, their deltas, and whether the top-1 ids agree.
"""
import json
import struct
import sys

import numpy as np


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    vocab = int(sys.argv[3]) if len(sys.argv) > 3 else 196608
    a = np.frombuffer(open(sys.argv[1], "rb").read(), dtype="<f4").reshape(-1, vocab)
    resp = json.load(open(sys.argv[2], encoding="utf-8"))
    steps = resp.get("completion_probabilities", [])
    worst = 0.0
    bad = 0
    for s, step in enumerate(steps):
        if s >= a.shape[0]:
            break
        r = a[s]
        m = r.max()
        logz = m + np.log(np.exp(r - m).sum())
        lp = [(t["id"], t["logprob"]) for t in step["top_logprobs"]]
        top1_ref = int(np.argmax(r))
        same = top1_ref == lp[0][0]
        bad += 0 if same else 1
        parts = []
        for i, l in lp:
            d = float(r[i] - logz) - l
            worst = max(worst, abs(d))
            parts.append("%d ref %+.4f llama %+.4f d %+.4f" % (i, float(r[i] - logz), l, d))
        print("step %d %s | %s" % (s, "top1 same" if same else "TOP1 DIFFERS (%d)" % top1_ref,
                                   " ; ".join(parts)))
    print("worst |delta| %.4f, top-1 mismatches %d" % (worst, bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
