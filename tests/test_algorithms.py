"""Independent numerical references and CLI checks for image-similarity."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from image_similarity.algorithms import search, inverted_index, descriptors
from image_similarity.data import image_demo, load_images
from image_similarity.runtime import BACKENDS, availability, normalize

from scipy.signal import convolve2d

def reference_descriptors(images, bins):
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    rows = []
    for image in images:
        im = image.astype(np.int32)
        gray = (77*im[..., 0] + 150*im[..., 1] + 29*im[..., 2]) // 256
        gx = convolve2d(gray, sobel, mode="valid")
        gy = convolve2d(gray, sobel.T, mode="valid")
        strength = np.minimum(bins-1, (np.abs(gx)+np.abs(gy))*bins//2041)
        hist = [np.bincount((im[..., c]*bins//256).ravel(), minlength=bins)/gray.size for c in range(3)]
        hist.append(np.bincount(strength.ravel(), minlength=bins)/strength.size)
        value = np.concatenate(hist)
        rows.append(value/np.linalg.norm(value))
    return np.asarray(rows)

def reference_topk(q, db, k):
    scores = q.astype(np.float64) @ db.astype(np.float64).T
    ids = np.stack([np.lexsort((np.arange(len(db)), -row))[:k] for row in scores])
    return ids, np.take_along_axis(scores, ids, axis=1)


class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)
        self.backends = [b for b in BACKENDS if availability(b)[0]]
        self.assertIn("cpu", self.backends, "Build the native library first")

    def test_streaming_search_odd_shapes_and_small_chunks(self):
        q, db = normalize(self.rng.normal(size=(5, 19))), normalize(self.rng.normal(size=(137, 19)))
        expected_ids, expected_scores = reference_topk(q, db, 11)
        for backend in self.backends:
            with self.subTest(backend=backend):
                ids, scores = search(q, db, 11, chunk=7, backend=backend, query_batch=2)
                np.testing.assert_array_equal(ids, expected_ids)
                np.testing.assert_allclose(scores, expected_scores, atol=2e-6)

    def test_tie_policy_zero_vectors_and_k_equals_database(self):
        db = np.array([[1, 0], [1, 0], [0, 0], [-1, 0], [0, 1], [1, 0]], np.float32)
        q = np.array([[1, 0], [0, 0]], np.float32)
        expected, scores = reference_topk(q, db, len(db))
        for backend in self.backends:
            with self.subTest(backend=backend):
                ids, actual = search(q, db, len(db), chunk=2, backend=backend)
                np.testing.assert_array_equal(ids, expected)
                np.testing.assert_allclose(actual, scores, atol=1e-6)

    def test_search_rejects_invalid_inputs(self):
        x = np.ones((3, 2), np.float32)
        for kwargs in ({"k": 0}, {"k": 4}, {"k": 1, "chunk": 0}, {"k": 1, "threads": 0}):
            with self.assertRaises(ValueError):
                search(x, x, **kwargs)
        with self.assertRaises(ValueError):
            search(x, np.ones((3, 4)), 1)
        with self.assertRaises(ValueError):
            search(np.full((1, 2), np.nan), x, 1)

    def test_inverted_index_empty_clusters_and_stable_grouping(self):
        labels = np.array([2, 0, 2, 0, 3], np.int32)
        offsets, permutation = inverted_index(labels, 5)
        np.testing.assert_array_equal(offsets, [0, 2, 2, 4, 5, 5])
        np.testing.assert_array_equal(permutation, [1, 3, 0, 2, 4])
        empty_offsets, empty_order = inverted_index([], 3)
        np.testing.assert_array_equal(empty_offsets, [0, 0, 0, 0])
        self.assertEqual(len(empty_order), 0)
        with self.assertRaises(RuntimeError):
            inverted_index([-1], 2)
        for backend in self.backends:
            if backend.startswith("cuda") or backend == "cublas":
                with self.subTest(backend=backend):
                    gpu_offsets, gpu_order = inverted_index(labels, 5, backend)
                    np.testing.assert_array_equal(gpu_offsets, offsets)
                    np.testing.assert_array_equal(np.sort(gpu_order), np.arange(len(labels)))
                    np.testing.assert_array_equal(labels[gpu_order], labels[permutation])
                    zero_offsets, zero_order = inverted_index([], 3, backend)
                    np.testing.assert_array_equal(zero_offsets, empty_offsets)
                    self.assertEqual(len(zero_order), 0)

    def test_descriptors_against_independent_convolution(self):
        images = self.rng.integers(0, 256, size=(5, 17, 19, 3), dtype=np.uint8)
        expected = reference_descriptors(images, 7)
        for backend in self.backends:
            with self.subTest(backend=backend):
                actual = descriptors(images, bins=7, backend=backend)
                np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_constant_images_have_known_descriptor(self):
        images = np.zeros((2, 3, 5, 3), np.uint8)
        expected = np.zeros((2, 32), np.float32)
        expected[:, [0, 8, 16, 24]] = 0.5
        for backend in self.backends:
            with self.subTest(backend=backend):
                np.testing.assert_allclose(descriptors(images, 8, backend), expected, atol=1e-6)

    def test_descriptor_input_validation(self):
        for images in (np.zeros((1, 2, 2, 3), np.uint8), np.zeros((1, 5, 5, 3), np.float32)):
            with self.assertRaises(ValueError):
                descriptors(images)

    def test_image_generation_and_self_match_exclusion(self):
        from image_similarity.app import match_images
        images, names, labels = image_demo(3, 17, 42)
        self.assertEqual(len(names), 9)
        _, matches = match_images(images, 8, "cpu", 2, 3)
        self.assertTrue(all(all(j != i for j, _ in row) for i, row in enumerate(matches)))
        self.assertTrue(all(labels[row[0][0]] == labels[i] for i, row in enumerate(matches)))

    def test_normalization_handles_zero_large_and_small_rows(self):
        np.testing.assert_allclose(normalize([[0, 0], [3, 4]]), [[0, 0], [0.6, 0.8]], atol=2e-6)
        np.testing.assert_allclose(normalize([[1e30, 1e30], [1e-40, 0]]),
                                   [[2**-0.5, 2**-0.5], [1, 0]], atol=2e-6)

    def test_cli_real_data_input(self):
        import json
        with tempfile.TemporaryDirectory(prefix="standalone-cli-") as directory:
            root = Path(directory)
            from PIL import Image
            images, names, _ = image_demo(3, 17)
            (root / "images").mkdir()
            for pixels, name in zip(images, names):
                Image.fromarray(pixels).save(root / "images" / (name+".png"))
            arguments = ["--images", str(root / "images"), "--size", "17"]
            out = root / "output"
            command = [sys.executable, str(ROOT / "run.py"), "--backends", "cpu", "--repeats", "1",
                       "--output", str(out), *arguments]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            record = json.loads((out / "results.json").read_text())["records"][0]
            self.assertEqual(record["status"], "ok")
            self.assertNotIn("pair_precision", record)
            self.assertNotIn("inlier_label_accuracy_after_matching", record)


if __name__ == "__main__":
    unittest.main(verbosity=2)
