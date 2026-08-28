"""Deterministic image-folder smoke fixture: 2 classes, 8 samples.

train/crimson/b.jpg carries EXIF orientation 6 so the bounded smoke
exercises the trainer's transpose-to-visual-orientation decode path.
No randomness: pixel content is a fixed function of position.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

SIZE = (64, 64)
CLASSES = {"crimson": (200, 30, 30), "cobalt": (30, 30, 200)}


def _image(base: tuple[int, int, int], salt: int) -> Image.Image:
    image = Image.new("RGB", SIZE)
    for x in range(SIZE[0]):
        for y in range(SIZE[1]):
            image.putpixel(
                (x, y),
                (
                    min(255, base[0] + (x * salt + y) % 17),
                    min(255, base[1] + (x + y) % 11),
                    min(255, base[2] + (x + y * salt) % 13),
                ),
            )
    return image


def generate(root: Path) -> int:
    root = Path(root)
    count = 0
    for split, per_class in (("train", 2), ("val", 1), ("test", 1)):
        for class_name, base in CLASSES.items():
            directory = root / split / class_name
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(per_class):
                image = _image(base, salt=2 + index)
                if split == "train" and class_name == "crimson" and index == 1:
                    exif = Image.Exif()
                    exif[0x0112] = 6  # rotate-90 tag: decode must transpose
                    image.save(
                        directory / "b.jpg", quality=98, subsampling=0, exif=exif
                    )
                else:
                    image.save(directory / f"{'ab'[index]}.png")
                count += 1
    return count


if __name__ == "__main__":
    import sys

    print(generate(Path(sys.argv[1])))
