# Parallel image similarity and near-duplicate candidates

Compute interpretable descriptors, rank similar images, and emit thresholded near-duplicate candidates with a visual contact sheet.

## Run

From this repository root, with its virtual environment activated and native library built:

```bash
python run.py --groups 32 --size 64 --backends cpu openmp mps
python run.py --images /path/to/photos --size 96 \
  --threshold 0.985 --output results/my-images
```

The demo creates source images and two variants per source: horizontal mirror and small pixel noise. It writes these inputs to `demo-images`. For a real directory, supported formats include PNG, JPEG, BMP, TIFF, and WebP. Images are converted to RGB and resized to a common square before timing. A corrupt image fails with its decode error rather than silently disappearing from the dataset.

## Pipeline

The descriptor has `4*bins` components (64 by default): a histogram for each RGB channel and a histogram of Sobel edge strengths. Integer luminance `(77R+150G+29B)//256` and L1 gradient magnitude keep histogram binning consistent across CPU, CUDA, and tensor backends. Sobel convolution covers valid interior pixels. Each histogram is divided by its contributing pixel count, then the complete descriptor is L2 normalized.

Exact retrieval finds up to five neighbors per image after removing self matches by ID. Thresholded lists are compacted using counts plus a prefix scan, and symmetric pairs are deduplicated and sorted. CUDA backends use atomic counts/scatter and CUB's device exclusive scan for compaction; other backends use the local CPU scan. GPU compaction preserves grouping, while within-group order is unspecified. CPU/OpenMP and optimized CUDA use per-image/per-tile parallelism, respectively. The optimized CUDA kernel loads grayscale tiles with a halo, accumulates block histograms, and merges them into each image descriptor. The tensor implementation uses convolution and scatter histograms on CPU or MPS.

Histograms can assign similar descriptors to different spatial arrangements. The output is a candidate report for review, not a definitive duplicate label. Thresholded pairs consider only the five retained neighbors per image; large duplicate groups can have unreported pairs.

## Results

`results/local/` contains:

* `results.json`, `timings.csv`, and `pipeline_latency.png` for measured descriptor + retrieval wall time.
* Per-backend descriptor arrays and compacted candidate indices.
* Per-backend pair CSVs and `ranked_matches.csv` with row IDs, filenames, scores, and ranks.
* `similar_pairs.png` showing the highest-scoring candidate pairs.

The demo reports pair precision/recall and top-1 same-source rate using its known transformation groups. Those metrics are omitted for unlabeled real directories. Every backend's descriptor is checked against C++; independent tests compare C++ against SciPy convolution and NumPy histograms.

```bash
python scripts/sweep.py --sizes 12 32 64 --threads 1 2 4
```

Sweep sizes are source groups (three images per group), and these sweeps time descriptor extraction only. Use the main runner to include retrieval.
