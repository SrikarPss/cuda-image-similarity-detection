#!/usr/bin/env python3
"""Parallel Image Similarity and Near-Duplicate Detection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from image_similarity.app import main

if __name__ == "__main__":
    main()
