"""
jigsaw_jeeves.model
===================

Computer-vision pipeline for providing piece-placement hints when solving
jigsaw puzzles.  The pipeline has three stages:

  1. Grid overlay  -- both the reference (box cover) and scrambled images
     are divided into an R-by-C grid of rectangular tiles.
  2. Feature extraction -- each tile is represented as a 513-dimensional
     vector: a 512-value normalized 3-D color histogram (8 bins per RGB
     channel) concatenated with a single edge-density scalar.
  3. Global assignment -- cosine similarity between every scrambled tile
     and every reference tile is computed in one vectorized operation, then
     the Hungarian algorithm finds the bijective mapping that maximises
     total similarity across all tile pairs simultaneously.

Public entry point
------------------
    solve(reference_image_filepath, scrambled_image_filepath, ...)

Internal helpers (not exported)
--------------------------------
    _infer_grid         -- estimate (R, C) from original image dimensions
    _load_image         -- load, background-suppress, and resize an image
    _image_to_matrix    -- slice a resized image into an R-by-C tile grid
    _extract_features   -- compute the 513-dim feature vector for one tile
    _predict            -- build similarity matrix and run Hungarian solver
    _save_results       -- write the ranked results dict to a text file
"""

import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_grid(image_path: str) -> tuple[int, int]:
    """Estimate a sensible (R, C) grid from the original image dimensions.

    The heuristic targets tiles of approximately 150x150 pixels, which
    empirically produces cells with enough visual content to be distinctive
    while remaining coarse enough to absorb small misalignments.

    Original (pre-resize) dimensions are used rather than the post-resize
    600x600 dimensions so that large-format photos (e.g. 4000x3000) get
    meaningful tile counts rather than a fixed 4x4.

    Parameters
    ----------
    image_path : str
        Path to the image file.  Must be readable by OpenCV.

    Returns
    -------
    tuple[int, int]
        (R, C) where R is the number of rows and C the number of columns.
        Both are at least 1.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(
            f"Could not load image at '{image_path}' for grid inference. "
            "Check that the file exists and is a valid image."
        )
    h, w = img.shape[:2]

    # Target ~150px per tile side; clamp to at least 1 in each dimension.
    r = max(1, round(h / 150))
    c = max(1, round(w / 150))
    return r, c


def _load_image(
    path: str,
    target_size: tuple[int, int] = (600, 600),
) -> np.ndarray:
    """Load an image, suppress the background, and resize to a fixed canvas.

    Background suppression removes the table surface that appears in
    photographs of physical puzzle pieces.  It works by:
      1. Applying a Gaussian blur to smooth out small pixel-level variations
         so the background reads as a near-uniform region.
      2. Converting to grayscale and running Otsu thresholding, which
         automatically finds the intensity cutoff that best separates the
         bright puzzle image (foreground) from the darker background.
      3. Multiplying the original image by the resulting binary mask to zero
         out all background pixels.

    Resizing both images to the same target canvas is essential so that
    the grid cells carved out of each image have identical pixel dimensions,
    which is required for the feature vectors to be directly comparable.

    Parameters
    ----------
    path : str
        Path to the image file.
    target_size : tuple[int, int], optional
        (width, height) to resize to.  Defaults to (600, 600).

    Returns
    -------
    np.ndarray
        RGB uint8 array of shape (target_size[1], target_size[0], 3).

    Raises
    ------
    ValueError
        If the file cannot be opened by OpenCV, or if the image is entirely
        black after background suppression (indicating a bad photo or
        insufficient foreground-background contrast).
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(
            f"Could not load image at '{path}'. "
            "Check that the file exists and is a valid image."
        )

    # OpenCV loads in BGR order; convert to RGB for all downstream operations.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Gaussian blur before thresholding: smooths pixel noise so the
    # background reads as a uniform region rather than a textured one,
    # which makes Otsu's threshold more reliable.
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)

    # Otsu thresholding automatically selects the intensity split that
    # maximises inter-class variance between foreground and background.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8)
    img_clean = img * mask[:, :, None]

    # Guard against photos where the background suppression wiped everything
    # (e.g. the puzzle pieces are the same colour as the table).
    if img_clean.max() == 0:
        raise ValueError(
            f"Image at '{path}' appears entirely black after background "
            "suppression. The photo may have insufficient contrast with "
            "the background -- try photographing on a differently coloured surface."
        )

    return cv2.resize(img_clean, target_size, interpolation=cv2.INTER_AREA)


def _image_to_matrix(
    image: np.ndarray,
    dimensions: tuple[int, int],
) -> list[list[np.ndarray]]:
    """Slice a resized image into an R-by-C grid of rectangular tiles.

    Integer division is used so the slicing is always exact on a 600x600
    canvas with sensible grid dimensions.  If the image dimensions are not
    perfectly divisible by R or C, a few pixels along the bottom or right
    edge are silently trimmed -- the loss is negligible in practice.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 array, typically the output of _load_image.
    dimensions : tuple[int, int]
        (R, C) -- number of rows and columns in the grid.

    Returns
    -------
    list[list[np.ndarray]]
        R-by-C nested list; each element is a uint8 tile array of shape
        (cell_h, cell_w, 3).
    """
    r, c = dimensions
    h, w = image.shape[:2]
    cell_h = h // r
    cell_w = w // c

    return [
        [
            image[i * cell_h:(i + 1) * cell_h,
                  j * cell_w:(j + 1) * cell_w]
            for j in range(c)
        ]
        for i in range(r)
    ]


def _extract_features(cell: np.ndarray) -> np.ndarray:
    """Compute a 513-dimensional feature vector for a single image tile.

    The vector has two components:

    Color histogram (512 values)
        A 3-D histogram over the R, G, and B channels with 8 bins each
        (8^3 = 512 bins total).  Each bin counts pixels with a particular
        combination of red, green, and blue values.  The histogram is
        normalized to sum to 1, making it a probability distribution.

        Normalization is the key design choice here: two photos of the
        same tile taken at different brightness levels produce similar
        histograms because the *shape* of the color distribution is
        preserved even when the overall exposure changes.  This directly
        addresses the cross-source distribution shift between the box cover
        photo and the smartphone photo of the scrambled pieces.

        Color histograms are fully rotation-invariant -- rotating a tile
        does not change which colors are present or in what proportions.

    Edge density scalar (1 value)
        The Canny edge detector is run on a grayscale version of the tile,
        producing a binary edge map (0 = no edge, 255 = edge detected).
        Edge density is the fraction of pixels that are edge pixels.

        This captures structural texture information that color alone
        misses: a cloudless sky tile and a foliage tile may have similar
        blue/green histograms but very different edge densities.

        A full 16-bin histogram of the binary Canny output was considered
        but rejected: because Canny only outputs 0 or 255, 14 of 16 bins
        would always be empty.  A single scalar is both correct and honest.

        Edge density is also rotation-invariant: the proportion of edge
        pixels in a tile does not depend on its orientation.

    Parameters
    ----------
    cell : np.ndarray
        RGB uint8 tile array of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Float64 array of shape (513,).
    """
    # --- Color histogram ---
    hist_color = cv2.calcHist(
        [cell], [0, 1, 2], None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256],
    ).flatten()
    # Normalize to sum to 1; add 1e-6 to prevent division by zero for
    # all-black tiles that can appear after background suppression.
    hist_color /= (np.sum(hist_color) + 1e-6)

    # --- Edge density ---
    gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    # Count edge pixels (value 255) and divide by total pixels.
    # 1e-6 guards against zero-area tiles (should not occur in practice).
    edge_density = np.array([np.sum(edges > 0) / (edges.size + 1e-6)])

    return np.concatenate([hist_color, edge_density])


def _predict(
    src_matrix: list[list[np.ndarray]],
    dest_matrix: list[list[np.ndarray]],
    top_k: int,
) -> dict[tuple[int, int], list[tuple[tuple[int, int], float]]]:
    """Match scrambled tiles to reference tiles using the Hungarian algorithm.

    The matching pipeline:
      1. Extract a 513-dim feature vector for every tile in both grids.
      2. L2-normalise all vectors so that their dot product equals cosine
         similarity.  Cosine similarity is chosen over Euclidean distance
         because it is scale-invariant: a tile photographed under brighter
         light has a larger feature-vector magnitude but the same direction,
         so its cosine similarity to the matching reference tile is
         unaffected.
      3. Compute the full N-by-N similarity matrix in a single matrix
         multiply (src_normed @ dest_normed.T).  This is substantially
         faster than a nested loop because NumPy dispatches the operation
         to optimised BLAS routines.
      4. Run the Hungarian algorithm (linear_sum_assignment) on the negated
         similarity matrix.  The algorithm solves a MINIMISATION problem,
         so negating converts "maximise total similarity" into
         "minimise total cost" -- the mathematically equivalent form.
         The Hungarian algorithm guarantees the globally optimal one-to-one
         assignment, which greedy matching cannot: a greedy approach either
         suffers from path dependence (sequential without replacement) or
         produces invalid many-to-one mappings (independent per tile).
      5. For each scrambled tile, return a ranked list of the top_k
         destination candidates by cosine similarity.  The globally
         assigned match is promoted to position 0 even if it was not the
         locally highest scorer -- the global assignment is trusted over
         any individual tile's local preference, because the Hungarian
         solver optimises total fit across all pairs simultaneously.

    Parameters
    ----------
    src_matrix : list[list[np.ndarray]]
        R-by-C nested list of scrambled tiles.
    dest_matrix : list[list[np.ndarray]]
        R-by-C nested list of reference (solved) tiles.
    top_k : int
        Number of candidate matches to return per scrambled tile.

    Returns
    -------
    dict[tuple[int, int], list[tuple[tuple[int, int], float]]]
        Maps each scrambled tile position (row, col) to a ranked list of
        up to top_k entries.  Each entry is ((dest_row, dest_col), score)
        where score is the cosine similarity in [0, 1].
    """
    r, c = len(src_matrix), len(src_matrix[0])
    n = r * c

    # Extract features for all tiles upfront to avoid redundant computation.
    src_features = np.array([
        _extract_features(src_matrix[i][j])
        for i in range(r) for j in range(c)
    ])  # shape: (n, 513)
    dest_features = np.array([
        _extract_features(dest_matrix[i][j])
        for i in range(r) for j in range(c)
    ])  # shape: (n, 513)

    # L2-normalise: after division, each row has unit norm, so the dot
    # product of two rows equals their cosine similarity.
    src_norms = np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6
    dest_norms = np.linalg.norm(dest_features, axis=1, keepdims=True) + 1e-6
    src_normed = src_features / src_norms
    dest_normed = dest_features / dest_norms

    # Full N-by-N cosine similarity matrix in one BLAS-dispatched multiply.
    sims = src_normed @ dest_normed.T  # shape: (n, n)

    # Hungarian algorithm: negate similarity to convert maximisation to
    # minimisation (linear_sum_assignment minimises by convention).
    row_ind, col_ind = linear_sum_assignment(-sims)
    assigned = dict(zip(row_ind.tolist(), col_ind.tolist()))

    results: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {}
    for src_idx in range(n):
        # Sort all destination tiles by descending cosine similarity to
        # this scrambled tile.
        sorted_dest = np.argsort(sims[src_idx])[::-1]
        matches = [
            ((int(j // c), int(j % c)), float(sims[src_idx, j]))
            for j in sorted_dest[:top_k]
        ]

        # Retrieve the globally assigned destination for this tile.
        assigned_j = assigned[src_idx]
        assigned_pos = (int(assigned_j // c), int(assigned_j % c))
        assigned_score = float(sims[src_idx, assigned_j])

        # Remove the assigned match if it already appears in the top-k list
        # (to avoid duplicates), then insert it at position 0.
        matches = [m for m in matches if m[0] != assigned_pos]
        matches.insert(0, (assigned_pos, assigned_score))
        matches = matches[:top_k]

        src_pos = (int(src_idx // c), int(src_idx % c))
        results[src_pos] = matches

    return results


def _save_results(results: dict, path: str) -> None:
    """Write the solve() results dict to a human-readable text file.

    Output format example::

        Cell (0, 0) best matches:
          -> (1, 2) (score=0.9821)
          -> (0, 3) (score=0.7043)

    Parameters
    ----------
    results : dict
        Output of _predict -- maps scrambled (row, col) to ranked list.
    path : str
        Full path to the output file to write.

    Raises
    ------
    ValueError
        If the parent directory of `path` does not exist.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        raise ValueError(
            f"Directory '{parent}' does not exist. "
            "Please create it before specifying save_to_file."
        )

    with open(path, "w") as f:
        for src_pos, matches in sorted(results.items()):
            f.write(f"Cell {src_pos} best matches:\n")
            for dest_pos, score in matches:
                f.write(f"  -> {dest_pos} (score={score:.4f})\n")
            f.write("\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve(
    reference_image_filepath: str,
    scrambled_image_filepath: str,
    grid: tuple[int, int] | None = None,
    top_k: int = 3,
    save_to_file: str | None = None,
) -> dict[tuple[int, int], list[tuple[tuple[int, int], float]]]:
    """Find the best matching destination for each tile in a scrambled puzzle.

    This is the single public entry point of jigsaw-jeeves.  It loads the
    reference (box cover) image and the scrambled image, divides both into
    the same R-by-C grid, extracts a 513-dimensional feature vector per
    tile, and finds the globally optimal bijective assignment using the
    Hungarian algorithm.

    The result tells you, for each position in the scrambled grid, where
    the tile most likely belongs in the solved reference grid -- narrowing
    a thousands-of-piece search to a small ranked candidate list.

    Parameters
    ----------
    reference_image_filepath : str
        Path to the puzzle box cover image (the completed, solved picture).
    scrambled_image_filepath : str
        Path to a photograph of the scrambled puzzle pieces arranged face-up
        in a neat rectangular grid on a flat surface.
    grid : tuple[int, int] or None, optional
        (rows, cols) matching the physical layout of pieces on the table.
        If None (default), the grid is inferred automatically from the
        original reference image dimensions, targeting tiles of roughly
        150x150 pixels.
    top_k : int, optional
        Number of candidate destination positions to return per scrambled
        tile (default 3).  The top-1 entry is always the globally optimal
        Hungarian assignment.
    save_to_file : str or None, optional
        If provided, results are written to this file path in addition to
        being returned as a Python dict.  The parent directory must exist.

    Returns
    -------
    dict[tuple[int, int], list[tuple[tuple[int, int], float]]]
        Maps each scrambled tile position (row, col) to a ranked list of
        up to top_k candidate solved positions.  Each entry is a tuple
        ((dest_row, dest_col), cosine_similarity_score).  The first entry
        is the globally optimal match; remaining entries are the next-best
        matches by cosine similarity.

    Raises
    ------
    ValueError
        If either image path cannot be opened, if `grid` is not a 2-tuple
        of positive integers, if `top_k` is not a positive integer, if
        `save_to_file`'s parent directory does not exist, or if either
        image is entirely black after background suppression.

    Examples
    --------
    >>> from jigsaw_jeeves import solve
    >>> results = solve("box_cover.jpg", "pieces_on_table.jpg", grid=(4, 4))
    >>> best_dest, score = results[(0, 0)][0]
    >>> print(f"Tile (0,0) belongs at {best_dest} (score={score:.3f})")
    """
    # ------------------------------------------------------------------
    # Input validation -- all checks before any image I/O so the caller
    # gets a clear error without waiting for image loading.
    # ------------------------------------------------------------------
    if not isinstance(reference_image_filepath, str) or not reference_image_filepath:
        raise ValueError("reference_image_filepath must be a non-empty string.")
    if not isinstance(scrambled_image_filepath, str) or not scrambled_image_filepath:
        raise ValueError("scrambled_image_filepath must be a non-empty string.")

    if not os.path.isfile(reference_image_filepath):
        raise ValueError(
            f"Reference image not found at '{reference_image_filepath}'. "
            "Check the file path."
        )
    if not os.path.isfile(scrambled_image_filepath):
        raise ValueError(
            f"Scrambled image not found at '{scrambled_image_filepath}'. "
            "Check the file path."
        )

    if grid is not None:
        if (
            not isinstance(grid, (tuple, list))
            or len(grid) != 2
            or not all(isinstance(x, int) and x > 0 for x in grid)
        ):
            raise ValueError(
                "grid must be a tuple of two positive integers (rows, cols), "
                f"e.g. (4, 4).  Got: {grid!r}"
            )
        grid = (int(grid[0]), int(grid[1]))

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(
            f"top_k must be a positive integer.  Got: {top_k!r}"
        )

    if save_to_file is not None:
        parent = os.path.dirname(os.path.abspath(save_to_file))
        if not os.path.isdir(parent):
            raise ValueError(
                f"Directory '{parent}' does not exist. "
                "Please create it before specifying save_to_file."
            )

    # ------------------------------------------------------------------
    # Grid inference (if not provided)
    # ------------------------------------------------------------------
    if grid is None:
        grid = _infer_grid(reference_image_filepath)

    # ------------------------------------------------------------------
    # Image loading and preprocessing
    # ------------------------------------------------------------------
    reference_img = _load_image(reference_image_filepath)
    scrambled_img = _load_image(scrambled_image_filepath)

    # ------------------------------------------------------------------
    # Grid slicing
    # ------------------------------------------------------------------
    reference_matrix = _image_to_matrix(reference_img, grid)
    scrambled_matrix = _image_to_matrix(scrambled_img, grid)

    # ------------------------------------------------------------------
    # Feature extraction, similarity computation, and global assignment
    # ------------------------------------------------------------------
    results = _predict(scrambled_matrix, reference_matrix, top_k)

    # ------------------------------------------------------------------
    # Optional file output
    # ------------------------------------------------------------------
    if save_to_file is not None:
        _save_results(results, save_to_file)

    return results
