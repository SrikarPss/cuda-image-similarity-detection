# Benchmark evidence

Archived CPU benchmark snapshot from development before repository separation. The standalone implementation retains the numerical algorithms and has separate [build/test evidence](../docs/VALIDATION.md).

## Measurements

Warm median wall time for descriptor extraction + exact search + self exclusion. Includes validation, allocation, transfers and result copies; excludes image loading, candidate compaction and report writing.

| Backend | Pipeline time | Speedup vs. sequential C++ |
|---|---:|---:|
| C++ / SIMD, 1 thread | 1.871 ms | 1.00× |
| C++ / OpenMP, 4 threads | 0.739 ms | 2.53× |
| PyTorch CPU, 4 threads | 3.190 ms | 0.59× |

Each successful row uses one warm-up and the median of three measured wall-clock samples. Full samples, input configuration, seed, quality metrics, and software versions are recorded in [results.json](results.json). Inputs are synthetic; these results do not establish real-world semantic or detection accuracy.

## Artifacts

- [Size/thread scaling](scaling.png)
- [Raw benchmark snapshot](results.json)
- [Descriptor/retrieval latency](pipeline_latency.png)
- [Candidate pair visualization](similar_pairs.png)
- [Candidate pair data](cpu-pairs.csv)

The scaling plot is a separate workload from the main latency table. Search sweeps use 32 queries with 64-dimensional embeddings; image sweeps measure descriptors only for three images per source group; clustering sweeps use 12 clusters and 10 updates. CPU thread counts are 1, 2, and 4.

## Reproduce

From the repository root after installing dependencies and building:

```bash
python run.py --backends cpu openmp torch-cpu --threads 4 --repeats 3 --output results/local
python scripts/report.py results/local
python scripts/sweep.py --threads 1 2 4
```

No CUDA/cuBLAS or MPS performance is reported in this snapshot. GPU execution must be measured on suitable hardware. MPI rank-one correctness was tested for the clustering project; multi-rank and multi-node performance remain unmeasured.
