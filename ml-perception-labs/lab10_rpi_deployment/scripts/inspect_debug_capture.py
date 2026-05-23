import argparse
from pathlib import Path

import cv2
import numpy as np


def summarize_image(path: Path) -> dict[str, float | int | str]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    brightness = float(np.mean(image))
    glare_pct = float(np.sum(image > 240) / image.size * 100.0)
    dark_pct = float(np.sum(image < 15) / image.size * 100.0)
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "brightness": brightness,
        "glare_pct": glare_pct,
        "dark_pct": dark_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect debug_raw.jpg/debug_crop.jpg from the Pi inference app.")
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()

    for image_path in args.images:
        stats = summarize_image(image_path)
        print(
            "{path}: {width}x{height}, brightness={brightness:.2f}, "
            "glare={glare_pct:.1f}%, dark={dark_pct:.1f}%".format(**stats)
        )
        if stats["glare_pct"] > 8.0:
            print("  Warning: glare is high enough to confuse reflective classes.")
        if stats["dark_pct"] > 25.0:
            print("  Warning: image is very dark; improve lighting or camera exposure.")


if __name__ == "__main__":
    main()
