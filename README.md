# Jigsaw Jeeves

A Python library that acts as a computer-vision assistant for solving jigsaw puzzles. Rather than solving the puzzle for you, it tells you where each piece most likely belongs, narrowing the search space to a small ranked candidate list so you can make progress when you are stuck.

---

## Installation

```bash
pip install jigsaw-jeeves
```

Requires Python 3.12 or above. 

Dependencies (`opencv-python`, `numpy`, `scipy`) are installed automatically.

---

## Prerequisites

To use Jigsaw Jeeves you need two images:

1. **Reference image** -- a photo of the puzzle box cover (the completed picture)
2. **Scrambled image** -- a photo of the puzzle pieces arranged face-up in a neat rectangular grid on a flat, contrasting surface

Both images should be reasonably well-lit and photographed from roughly overhead. The grid dimensions you pass to `solve()` must match how you physically arranged the pieces before photographing.

---

## Quick Start

```python
from jigsaw_jeeves import solve

results = solve(
    reference_image_filepath="box_cover.jpg",
    scrambled_image_filepath="pieces_on_table.jpg",
    grid=(20, 25),       # rows x cols matching how you arranged the pieces
    top_k=3,             # number of candidate positions to return per piece
    save_to_file="solution.txt",  # optional: also write results to a file
)

# results maps each scrambled tile position to a ranked list of candidates.
for pos, candidates in results.items():
    best_dest, score = candidates[0]
    print(f"Piece at {pos} most likely belongs at {best_dest}  (similarity: {score:.3f})")
```

---

## `solve()` -- full signature

```python
solve(
    reference_image_filepath: str,
    scrambled_image_filepath: str,
    grid: tuple[int, int] | None = None,
    top_k: int = 3,
    save_to_file: str | None = None,
) -> dict
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `reference_image_filepath` | `str` | Path to the puzzle box cover image (the solved reference). |
| `scrambled_image_filepath` | `str` | Path to a photograph of the scrambled pieces arranged face-up in a rectangle on a flat surface. |
| `grid` | `tuple[int, int]` or `None` | `(rows, cols)` matching the physical layout of pieces. If `None`, inferred automatically from the reference image dimensions (targeting tiles of ~150×150 px). |
| `top_k` | `int` | Number of candidate destination positions to return per tile (default `3`). |
| `save_to_file` | `str` or `None` | If provided, results are written to this file path. The parent directory must exist. The dict is always returned regardless. |

### Returns

A `dict` mapping each scrambled tile position `(row, col)` to a ranked list of up to `top_k` candidate solved positions. Each list entry is `((dest_row, dest_col), cosine_similarity_score)`. The first entry is the globally optimal assignment (via the Hungarian algorithm); remaining entries are the next-best matches by cosine similarity.

### Raises

`ValueError` with a human-friendly message if:
- Either image path does not exist or cannot be opened
- `grid` is not a 2-tuple of positive integers
- `top_k` is not a positive integer
- `save_to_file`'s parent directory does not exist
- Either image is entirely black after background suppression (insufficient contrast with background)

---

## How It Works

- The pipeline overlays an R-by-C grid on both images and treats each grid cell as the unit of comparison.
- Each cell is represented as a 513-dimensional feature vector: a normalized 3-D RGB color histogram (512 values) and a single edge-density scalar.
- Cosine similarity is used to measure how closely a scrambled tile matches each reference tile.
- The Hungarian algorithm finds the globally optimal bijective assignment, guaranteeing that every scrambled tile is matched to a unique reference position.

---

## License

MIT
