import numpy as np
from .runtime import normalize
from pathlib import Path
from PIL import Image, ImageDraw


def image_demo(groups=24, size=64, seed=42):
    if groups < 2 or size < 16:
        raise ValueError("Image demo needs at least two groups and size>=16")
    rng = np.random.default_rng(seed)
    images, names, labels = [], [], []
    for group in range(groups):
        color = tuple(int(v) for v in rng.integers(20, 235, size=3))
        canvas = Image.new("RGB", (size, size), color)
        draw = ImageDraw.Draw(canvas)
        for _ in range(8):
            x0, y0 = rng.integers(0, size - 4, size=2)
            x1, y1 = rng.integers(x0 + 1, size), rng.integers(y0 + 1, size)
            fill = tuple(int(v) for v in rng.integers(0, 256, size=3))
            if rng.random() > 0.5:
                draw.rectangle((int(x0), int(y0), int(x1), int(y1)), fill=fill)
            else:
                draw.ellipse((int(x0), int(y0), int(x1), int(y1)), fill=fill)
        base = np.asarray(canvas).copy()
        variants = [base, np.fliplr(base).copy(),
                    np.clip(base.astype(np.int16) + rng.integers(-3, 4, size=base.shape), 0, 255).astype(np.uint8)]
        for variant, pixels in enumerate(variants):
            images.append(pixels)
            names.append(f"group_{group:03d}_variant_{variant}")
            labels.append(group)
    return np.stack(images), names, np.asarray(labels)


def load_images(directory, size=64):
    if size < 3:
        raise ValueError("Image size must be >=3")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    files = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in extensions and p.is_file())
    images = []
    for path in files:
        with Image.open(path) as im:
            images.append(np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)).copy())
    if len(images) < 2:
        raise ValueError("Provide a directory with at least two readable images")
    return np.stack(images), [str(p.relative_to(directory)) for p in files], None
