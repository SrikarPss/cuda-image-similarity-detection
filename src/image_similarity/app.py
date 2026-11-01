import argparse
import numpy as np
from PIL import Image, ImageDraw
from .algorithms import descriptors, inverted_index, search
from .data import image_demo, load_images
from .runtime import (add_common_arguments, availability, benchmark, metadata, output_directory,
                      timing_plot, validate_common, write_csv, write_json)


def match_images(images, bins, backend, threads, chunk):
    features = descriptors(images, bins, backend, threads)
    ids, scores = search(features, features, min(6, len(images)), chunk, backend, threads)
    rows = []
    for i, (indices, similarities) in enumerate(zip(ids, scores)):
        # Exclude self by row identity; identical descriptors may place self at any tied rank.
        rows.append([(int(j), float(s)) for j, s in zip(indices, similarities) if j != i][:5])
    return features, rows


def pair_report(images, names, pairs, path):
    thumb, text_height, count = 96, 30, min(8, len(pairs))
    if not count:
        return
    canvas = Image.new("RGB", (2*thumb + 280, count*(thumb + text_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, pair in enumerate(pairs[:count]):
        i, j, score = pair["left_id"], pair["right_id"], pair["similarity"]
        y = row*(thumb + text_height)
        canvas.paste(Image.fromarray(images[i]).resize((thumb, thumb)), (0, y))
        canvas.paste(Image.fromarray(images[j]).resize((thumb, thumb)), (thumb, y))
        draw.text((2*thumb + 8, y + 8), f"cosine = {score:.5f}", fill="black")
        draw.text((2*thumb + 8, y + 28), names[i][:36], fill="black")
        draw.text((2*thumb + 8, y + 48), names[j][:36], fill="black")
    canvas.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parallel color/edge image similarity and near-duplicate candidates")
    add_common_arguments(parser, "local")
    parser.add_argument("--images", type=str, help="Directory of actual images; otherwise generate a controlled demo")
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.985)
    parser.add_argument("--chunk", type=int, default=2048)
    args = parser.parse_args(argv)
    validate_common(args)
    if not 2 <= args.bins <= 32 or not -1 <= args.threshold <= 1 or args.chunk < 1:
        parser.error("bins must be 2..32, threshold -1..1, and chunk positive")
    images, names, truth = load_images(args.images, args.size) if args.images else image_demo(args.groups, args.size, args.seed)
    out = output_directory(args.output)
    if truth is not None:
        demo_dir = output_directory(out / "demo-images")
        for pixels, name in zip(images, names):
            Image.fromarray(pixels).save(demo_dir / (name + ".png"))
    reference = descriptors(images, args.bins, "cpu", args.threads)
    records, report_pairs, reference_matches = [], [], None
    for backend in dict.fromkeys(args.backends):
        ok, reason = availability(backend)
        if not ok:
            records.append({"backend": backend, "status": "skipped", "reason": reason})
            print(f"{backend}: skipped ({reason})", flush=True)
            continue
        try:
            (features, matches), elapsed = benchmark(
                lambda: match_images(images, args.bins, backend, args.threads, args.chunk), backend, args.repeats)
            error = float(np.max(np.abs(features - reference)))
            if error > 3e-5:
                raise RuntimeError(f"Descriptor error {error} exceeds tolerance")
            # Count + prefix scan compacts the variable-length per-query candidate lists.
            owners, candidates = [], []
            for i, row in enumerate(matches):
                for j, score in row:
                    if score >= args.threshold:
                        owners.append(i)
                        candidates.append((i, j, score))
            offsets, permutation = inverted_index(np.asarray(owners, dtype=np.int32), len(images), backend)
            pairs = {}
            for i, j, score in candidates:
                left, right = sorted((i, j))
                pairs[(left, right)] = {"left_id": left, "right_id": right, "left": names[left],
                                       "right": names[right], "similarity": score}
            pair_rows = sorted(pairs.values(), key=lambda row: (-row["similarity"], row["left_id"], row["right_id"]))
            record = {"backend": backend, "status": "ok", **elapsed, "images_per_second": len(images)/(elapsed["median_ms"]/1000),
                      "max_descriptor_error": error, "candidate_pairs": len(pair_rows)}
            if truth is not None:
                true_pairs = sum(truth[r["left_id"]] == truth[r["right_id"]] for r in pair_rows)
                possible = sum(int(count)*(int(count)-1)//2 for count in np.bincount(truth))
                record.update({"pair_precision": float(true_pairs/len(pair_rows)) if pair_rows else 0.0,
                               "pair_recall": float(true_pairs/possible) if possible else 0.0,
                               "top1_same_group_rate": float(np.mean([truth[i] == truth[row[0][0]] for i, row in enumerate(matches)]))})
            records.append(record)
            np.savez(out / f"{backend}.npz", descriptors=features, candidate_offsets=offsets,
                     candidate_permutation=permutation, candidate_ids=np.asarray([j for _, j, _ in candidates], dtype=np.int32))
            write_csv(out / f"{backend}-pairs.csv", pair_rows)
            if reference_matches is None:
                reference_matches, report_pairs = matches, pair_rows
            print(f"{backend}: {elapsed['median_ms']:.3f} ms; {len(pair_rows)} candidate pairs; error={error:.2g}", flush=True)
        except Exception as exc:
            records.append({"backend": backend, "status": "error", "reason": str(exc)})
            print(f"{backend}: ERROR {exc}", flush=True)
    if report_pairs:
        pair_report(images, names, report_pairs, out / "similar_pairs.png")
    if reference_matches is not None:
        write_csv(out / "ranked_matches.csv", [{"query_id": i, "query": names[i], "rank": rank+1,
                                                  "match_id": j, "match": names[j], "similarity": score}
                                                 for i, row in enumerate(reference_matches) for rank, (j, score) in enumerate(row)])
    write_json(out / "results.json", {"project": "image-similarity", "environment": metadata(),
              "source": str(args.images) if args.images else "synthetic shapes with mirror and small-noise variants",
              "images": len(images), "size": args.size, "bins": args.bins, "threshold": args.threshold,
              "seed": args.seed, "threads": args.threads,
              "timing_scope": "Warm median wall time for descriptor extraction + exact search + self exclusion. "
                              "Includes validation, allocation, transfers and result copies; excludes image loading, "
                              "candidate compaction and report writing.",
              "limitations": "Histogram candidates require review. Spatial arrangements can share descriptors. "
                              "Only the top five neighbors per image are considered for thresholded pairs.", "records": records})
    write_csv(out / "timings.csv", records)
    timing_plot(out / "pipeline_latency.png", records, "Image descriptors + retrieval: warm wall time")
    print("Results:", out)
    if not any(r["status"] == "ok" for r in records) or any(r["status"] == "error" for r in records):
        raise SystemExit(1)
