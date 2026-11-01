#!/usr/bin/env python3
"""Measure image-similarity problem-size and thread scaling."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from image_similarity.algorithms import descriptors
from image_similarity.data import image_demo
from image_similarity.runtime import BACKENDS, availability, benchmark, metadata, output_directory, write_csv, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sizes", nargs="+", type=int, default=[12, 32, 64])
parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["cpu", "openmp"])
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--output", type=Path, default=ROOT / "results/scaling")
args = parser.parse_args()
if args.dim < 2 or args.repeats < 1 or min(args.threads) < 1 or min(args.sizes) < 12:
    parser.error("dim>=2, positive repeats/threads, and sizes>=12 are required")
out = output_directory(args.output)
records = []
for size in args.sizes:
    images, _, _ = image_demo(size, 64, 42)
    operation = lambda b, t: descriptors(images, 16, b, t)
    reference = operation("cpu", 1)
    for backend in dict.fromkeys(args.backends):
        ok, reason = availability(backend)
        if not ok:
            records.append({"size": size, "backend": backend, "status": "skipped", "reason": reason})
            continue
        for threads in ([1] if backend == "cpu" else args.threads):
            import torch
            torch.set_num_threads(threads)
            result, elapsed = benchmark(lambda: operation(backend, threads), backend, args.repeats)
            np.testing.assert_allclose(result, reference, atol=3e-5)
            records.append({"size": size, "backend": backend, "threads": threads, "status": "ok", **elapsed})
            print(f"size={size} {backend} threads={threads}: {elapsed['median_ms']:.3f} ms", flush=True)
write_json(out / "results.json", {"project": "image-similarity", "environment": metadata(), "dim": args.dim,
           "timing_scope": "Warm API wall time after one warm-up. Descriptor extraction only: 64x64 RGB, 16 bins per histogram; sizes are source groups, three images each.", "records": records})
write_csv(out / "timings.csv", records)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))
series = sorted({(row["backend"], row["threads"]) for row in records if row["status"] == "ok"})
for backend, threads in series:
    rows = sorted([row for row in records if row["status"] == "ok" and row["backend"] == backend and row["threads"] == threads], key=lambda row: row["size"])
    ax.plot([row["size"] for row in rows], [row["median_ms"] for row in rows], "o-", label=f"{backend}, {threads} thread(s)")
ax.set(xlabel="Source image groups", ylabel="Wall-clock time (ms)", title="image-similarity: size and thread scaling")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(out / "scaling.png", dpi=160)
plt.close(fig)
print("Results:", out)
