from __future__ import annotations

import numpy as np
from .runtime import BYTE, DOUBLE, INT, cuda_mode, matrix, native, normalize, pointer, require


def search(queries, database, k=10, chunk=4096, backend="cpu", threads=4, query_batch=128):
    """Exact cosine top-k for unit rows; ties prefer smaller database row IDs.

    Copies, allocation, and validation are included when callers time this API.
    Query batching and database streaming bound temporary score storage.
    """
    require(backend)
    q, db = matrix(queries, "queries"), matrix(database, "database")
    if q.shape[1] != db.shape[1] or not 1 <= k <= len(db):
        raise ValueError("Dimensions must match and 1 <= k <= database size")
    if chunk < 1 or threads < 1 or query_batch < 1:
        raise ValueError("chunk, threads, and query_batch must be positive")
    indices = np.empty((len(q), k), np.int32)
    values = np.empty((len(q), k), np.float32)
    lib = native()
    for start in range(0, len(q), query_batch):
        part = q[start:start + query_batch]
        ids, scores = indices[start:start + len(part)], values[start:start + len(part)]
        if backend in ("cpu", "openmp"):
            status = lib.cpu.hpc_search(pointer(part), pointer(db), len(part), len(db), db.shape[1],
                                        k, chunk, threads, int(backend == "openmp"), pointer(ids, INT), pointer(scores))
            lib.check(status)
        elif backend.startswith("cuda") or backend == "cublas":
            status = lib.cuda.hpc_cuda_search(pointer(part), pointer(db), len(part), len(db), db.shape[1],
                                              k, chunk, cuda_mode(backend), pointer(ids, INT), pointer(scores))
            lib.check(status, cuda=True)
        else:
            import torch
            device = "mps" if backend == "mps" else "cpu"
            with torch.inference_mode():
                tq = torch.as_tensor(part, device=device)
                best = torch.empty((len(part), 0), device=device)
                best_ids = torch.empty((len(part), 0), device=device, dtype=torch.int64)
                for offset in range(0, len(db), chunk):
                    td = torch.as_tensor(db[offset:offset + chunk], device=device)
                    s = tq @ td.T
                    candidate_ids = torch.arange(offset, offset + len(td), device=device).expand(len(part), -1)
                    joined = torch.cat((best, s), dim=1)
                    joined_ids = torch.cat((best_ids, candidate_ids), dim=1)
                    # Earlier chunks have smaller IDs; stable score sorting preserves the tie policy.
                    order = torch.argsort(joined, dim=1, descending=True, stable=True)[:, :k]
                    best = torch.gather(joined, 1, order)
                    best_ids = torch.gather(joined_ids, 1, order)
                scores[:] = best.cpu().numpy()
                ids[:] = best_ids.cpu().numpy()
    return indices, values


def inverted_index(labels, clusters, backend="cpu"):
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    if labels.ndim != 1 or clusters < 1:
        raise ValueError("Expected 1D labels and a positive cluster count")
    offsets, permutation = np.empty(clusters + 1, np.int32), np.empty(len(labels), np.int32)
    lib = native()
    if backend.startswith("cuda") or backend == "cublas":
        require(backend)
        lib.check(lib.cuda.hpc_cuda_inverted_index(pointer(labels, INT), len(labels), clusters,
                                                   pointer(offsets, INT), pointer(permutation, INT)), cuda=True)
        return offsets, permutation
    if lib.cpu is None:
        raise RuntimeError(lib.cpu_error)
    lib.check(lib.cpu.hpc_inverted_index(pointer(labels, INT), len(labels), clusters,
                                        pointer(offsets, INT), pointer(permutation, INT)))
    return offsets, permutation


def descriptors(images, bins=16, backend="cpu", threads=4):
    require(backend)
    images = np.asarray(images)
    if (images.dtype != np.uint8 or images.ndim != 4 or images.shape[3] != 3
            or not images.shape[0] or min(images.shape[1:3]) < 3 or not 2 <= bins <= 32 or threads < 1):
        raise ValueError("Expected nonempty uint8 NHWC RGB images >=3x3, bins=2..32, threads>=1")
    images = np.ascontiguousarray(images)
    n, h, w, _ = images.shape
    if images.size > np.iinfo(np.int32).max:
        raise ValueError("Image batch exceeds supported int32 element count; split the input")
    out = np.empty((n, 4*bins), np.float32)
    lib = native()
    if backend in ("cpu", "openmp"):
        status = lib.cpu.hpc_descriptors(pointer(images, BYTE), n, h, w, bins, threads,
                                         int(backend == "openmp"), pointer(out))
        lib.check(status)
    elif backend.startswith("cuda") or backend == "cublas":
        status = lib.cuda.hpc_cuda_descriptors(pointer(images, BYTE), n, h, w, bins,
                                               cuda_mode(backend), pointer(out))
        lib.check(status, cuda=True)
    else:
        import torch
        import torch.nn.functional as functional
        device = "mps" if backend == "mps" else "cpu"
        with torch.inference_mode():
            x = torch.as_tensor(images.astype(np.int32), device=device)
            gray = ((77*x[..., 0] + 150*x[..., 1] + 29*x[..., 2]) // 256).float()[:, None]
            kernels = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], device=device).float()[:, None]
            edges = functional.conv2d(gray, kernels).abs().sum(dim=1)
            edge_bins = torch.clamp((edges * bins / 2041).long(), max=bins - 1).reshape(n, -1)
            histograms = []
            for c in range(3):
                buckets = ((x[..., c] * bins) // 256).long().reshape(n, -1)
                hist = torch.zeros((n, bins), device=device)
                hist.scatter_add_(1, buckets, torch.ones_like(buckets, dtype=torch.float32))
                histograms.append(hist / (h*w))
            hist = torch.zeros((n, bins), device=device)
            hist.scatter_add_(1, edge_bins, torch.ones_like(edge_bins, dtype=torch.float32))
            histograms.append(hist / ((h-2)*(w-2)))
            out[:] = functional.normalize(torch.cat(histograms, dim=1), dim=1).cpu().numpy()
    return out
