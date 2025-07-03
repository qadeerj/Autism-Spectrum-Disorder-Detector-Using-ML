# helper/preprocess.py
# --------------------------------------------------
# Pipeline to keep **exactly ≤15** most‑informative slices
# from every 3‑D MRI volume:
#   1. For each plane (sagittal, coronal, transverse) keep the
#      top 16 % slices by informativeness (minimum 1).
#   2. From that pooled set select the global top 15.
#   3. Tight‑crop, resize to 145×145, and save as .npy.
# --------------------------------------------------

import os
import re
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from typing import List, Set

# -------------------------- helpers --------------------------- #

def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """Scale intensities to [0, 1]."""
    max_val = img.max()
    return img / max_val if max_val > 0 else img


def apply_smoothing(img: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Light Gaussian smoothing to reduce speckle noise."""
    return gaussian_filter(img, sigma=sigma)


def extract_slices(volume: np.ndarray):
    """Yield tuples (plane, idx, slice_array)."""
    # sagittal: planes along the x‑axis
    for i in range(volume.shape[0]):
        yield "sagittal", i, volume[i, :, :]
    # coronal: planes along the y‑axis
    for j in range(volume.shape[1]):
        yield "coronal", j, volume[:, j, :]
    # transverse / axial: planes along the z‑axis
    for k in range(volume.shape[2]):
        yield "transverse", k, volume[:, :, k]


def calculate_informativeness(slice_arr: np.ndarray) -> float:
    """Simple foreground metric: 1 − (proportion of zero pixels)."""
    return 1.0 - (np.count_nonzero(slice_arr == 0) / slice_arr.size)


# -------------------- selection logic ------------------------- #

def select_percent_then_global(slice_paths: List[str],
                               top_percent: float = 0.16,
                               global_k: int = 15) -> Set[str]:
    """Implements the two‑stage slice selection scheme.

    Parameters
    ----------
    slice_paths : list of str
        All slice .npy paths from one volume.
    top_percent : float, optional
        Fraction of slices to keep **per plane** before global re‑ranking.
    global_k : int, optional
        Final number of slices to retain across all planes.

    Returns
    -------
    keep : set of str
        Paths to slices that survive both stages.
    """
    # 1️⃣ Bucket by plane
    plane_buckets = {"sagittal": [], "coronal": [], "transverse": []}
    plane_regex = re.compile(r".*_(sagittal|coronal|transverse)_[0-9]+\.npy$")
    for p in slice_paths:
        m = plane_regex.match(p)
        if m:
            plane_buckets[m.group(1)].append(p)

    # 2️⃣ Stage 1 — per‑plane pruning
    stage1_pool = []  # list of (score, path)
    for plane, paths in plane_buckets.items():
        if not paths:
            continue
        scores = [(calculate_informativeness(np.load(p)), p) for p in paths]
        n_keep = max(1, round(len(scores) * top_percent))
        stage1_pool.extend(sorted(scores, key=lambda x: x[0], reverse=True)[:n_keep])

    # 3️⃣ Stage 2 — global re‑ranking
    stage1_pool.sort(key=lambda x: x[0], reverse=True)
    keep = {p for _, p in stage1_pool[:global_k]}
    return keep


def crop_slice(slice_arr: np.ndarray) -> np.ndarray:
    """Tight crop around the non‑zero foreground."""
    coords = np.argwhere(slice_arr > 0)
    if coords.size == 0:
        return slice_arr  # leave blank slices untouched
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return slice_arr[y0 : y1 + 1, x0 : x1 + 1]


# -------------------------- main io --------------------------- #

def preprocess_images(
    input_dir: str,
    output_dir: str,
    out_size=(145, 145),
    top_percent: float = 0.16,
    global_k: int = 15,
):
    """Full preprocessing pipeline.

    Steps
    -----
    1.  Load every NIfTI file in *input_dir*.
    2.  Normalise → smooth.
    3.  Dump **all** slices as .npy.
    4.  Run two‑stage slice selection (16 % per plane → top 15 global).
    5.  Crop + resize survivors and remove the rest.
    """

    os.makedirs(output_dir, exist_ok=True)
    nii_pattern = re.compile(r".*\.nii(\.gz)?$", re.IGNORECASE)

    for fn in tqdm(os.listdir(input_dir), desc="Volumes"):
        if not nii_pattern.match(fn):
            continue

        subj = os.path.splitext(fn)[0]
        vol = nib.load(os.path.join(input_dir, fn)).get_fdata()
        vol = apply_smoothing(normalize_intensity(vol))

        # ① Dump every slice to disk
        slice_paths: List[str] = []
        for plane, idx, slc in extract_slices(vol):
            path = os.path.join(output_dir, f"{subj}_{plane}_{idx}.npy")
            np.save(path, slc.astype(np.float32))
            slice_paths.append(path)

        # ② Two‑stage selection
        keep = select_percent_then_global(
            slice_paths, top_percent=top_percent, global_k=global_k
        )

        # ③ Delete rejected slices
        for p in slice_paths:
            if p not in keep:
                os.remove(p)

        # ④ Crop + resize survivors
        for p in keep:
            slc = np.load(p)
            slc = crop_slice(slc)
            slc = resize(slc, out_size, order=3, preserve_range=True)
            np.save(p, slc.astype(np.float32))


# -------------------------- cli --------------------------- #
if __name__ == "__main__":
    preprocess_images(
        input_dir="PATH/TO/NIFTI_FILES",    # ← update
        output_dir="PATH/TO/OUTPUT_ROOT",   # ← update
        out_size=(145, 145),
        top_percent=0.16,   # 16 % per plane
        global_k=15,        # final cap
    )
