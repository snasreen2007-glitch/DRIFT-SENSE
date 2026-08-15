"""
dataset_generator.py
=======================
Implements Document Section 4 (Dataset Generation).

Each generated sample matches EVERY field the document requires:
  - Reference wafer image (crop from a synthetic "wafer" image)
  - Search image (perturbed full image)
  - Ground-truth coordinate
  - Artificially introduced navigation offset
  - Scale factor
  - Rotation parameter
  - Illumination variation
  - Blur / noise level
  - Optional occlusion
  - Ground-truth correction vector

I' = T_occlusion( T_noise( T_blur( T_illum( T_geom(I) ) ) ) )

Run standalone:  python src/dataset_generator.py --n 30
"""

import argparse
import json
import os
import numpy as np
import cv2


def _make_synthetic_wafer(size=1000, seed=0):
    """Creates a repetitive, wafer-like grid pattern with some unique
    'defect' motifs -- repetitive regions stress-test Top-K candidate
    selection (Section 3.5), unique motifs give clean, distinctive
    localization targets (representative of real inspection sites such
    as alignment marks or defect coordinates).

    Returns (image, list_of_motif_centers) so the dataset generator can
    deliberately choose most reference patches to be DISTINCTIVE
    (realistic) while still sampling some purely repetitive patches on
    purpose, as intentional stress/failure cases.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 200, dtype=np.uint8)

    # repetitive die grid
    step = 40
    for y in range(0, size, step):
        cv2.line(img, (0, y), (size, y), 150, 1)
    for x in range(0, size, step):
        cv2.line(img, (x, 0), (x, size), 150, 1)

    # repeated periodic structures (challenging / ambiguous regions)
    for y in range(20, size, step):
        for x in range(20, size, step):
            cv2.rectangle(img, (x - 8, y - 8), (x + 8, y + 8), 100, -1)

    # unique motifs scattered around (used as reference-patch anchors)
    motif_centers = []
    for i in range(30):
        cx, cy = rng.integers(100, size - 100, size=2)
        cv2.circle(img, (int(cx), int(cy)), int(rng.integers(12, 22)), int(rng.integers(30, 90)), -1)
        cv2.putText(img, str(i), (int(cx) - 10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, 0, 1, cv2.LINE_AA)
        motif_centers.append((int(cx), int(cy)))

    noise = rng.normal(0, 4, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, motif_centers


def _apply_illumination(img: np.ndarray, factor: float, bias: int) -> np.ndarray:
    out = img.astype(np.float32) * factor + bias
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return img
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _apply_noise(img: np.ndarray, sigma: float, rng) -> np.ndarray:
    if sigma <= 0:
        return img
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _apply_occlusion(img: np.ndarray, rng, prob: float = 0.3) -> np.ndarray:
    if rng.random() > prob:
        return img
    out = img.copy()
    h, w = out.shape
    ow, oh = rng.integers(20, 60, size=2)
    ox, oy = rng.integers(0, w - ow), rng.integers(0, h - oh)
    out[oy:oy + oh, ox:ox + ow] = int(rng.integers(0, 255))
    return out


def generate_sample(base_img: np.ndarray, motif_centers: list, sample_id: int, rng,
                     ref_size: int = 64, search_size: int = 320,
                     repetitive_stress_prob: float = 0.15) -> dict:
    """Models a realistic navigation-error-recovery scenario (Document
    Section 1/2): the stage is already CLOSE to the target -- we only
    need to recover a small residual drift/misalignment, not perform a
    blind search over the entire wafer. So the search image is a LOCAL
    crop around the nominal target position, not the whole canvas.

    Most samples are anchored on a distinctive motif (representative of
    a real inspection target, e.g. an alignment mark). A minority are
    anchored on the purely repetitive grid on purpose -- these are the
    intentional hard/failure cases required by Section 7.2.
    """
    h, w = base_img.shape
    half_search = search_size // 2

    is_repetitive_stress_case = rng.random() < repetitive_stress_prob
    if motif_centers and not is_repetitive_stress_case:
        cx, cy = motif_centers[rng.integers(0, len(motif_centers))]
    else:
        cx = int(rng.integers(half_search + 20, w - half_search - 20))
        cy = int(rng.integers(half_search + 20, h - half_search - 20))

    # clamp so the local search window fits inside the base image
    cx = int(np.clip(cx, half_search + 5, w - half_search - 5))
    cy = int(np.clip(cy, half_search + 5, h - half_search - 5))

    half_ref = ref_size // 2
    reference = base_img[cy - half_ref:cy + half_ref, cx - half_ref:cx + half_ref].copy()

    # local search window, nominal target sits at its center
    local = base_img[cy - half_search:cy + half_search, cx - half_search:cx + half_search].copy()
    lh, lw = local.shape
    nominal = np.array([lw / 2, lh / 2, 1.0])

    # --- SMALL perturbations: this is residual drift, not a global blind search ---
    scale = float(rng.uniform(0.92, 1.08))
    rotation = float(rng.uniform(-4, 4))
    illum_factor = float(rng.uniform(0.7, 1.3))
    illum_bias = int(rng.integers(-25, 25))
    blur_k = int(rng.choice([1, 1, 3, 5]))
    noise_sigma = float(rng.uniform(0, 6))
    offset_x = float(rng.uniform(-25, 25))
    offset_y = float(rng.uniform(-25, 25))

    M = cv2.getRotationMatrix2D((lw / 2, lh / 2), rotation, scale)
    # bake the navigation-drift offset directly into the affine transform
    # so the image CONTENT actually shifts by (offset_x, offset_y) --
    # otherwise the ground-truth label would claim a shift that never
    # happened in the pixels, which the detector could never correctly find.
    M[0, 2] += offset_x
    M[1, 2] += offset_y

    search = cv2.warpAffine(local, M, (lw, lh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    search = _apply_illumination(search, illum_factor, illum_bias)
    search = _apply_blur(search, blur_k)
    search = _apply_noise(search, noise_sigma, rng)
    search = _apply_occlusion(search, rng)

    tgx, tgy = M @ nominal
    gt_x = float(np.clip(tgx, half_ref, lw - half_ref))
    gt_y = float(np.clip(tgy, half_ref, lh - half_ref))

    return {
        "sample_id": sample_id,
        "reference": reference,
        "search": search,
        "ground_truth": {"x": gt_x, "y": gt_y},
        "params": {
            "scale": scale,
            "rotation_deg": rotation,
            "illumination_factor": illum_factor,
            "illumination_bias": illum_bias,
            "blur_ksize": blur_k,
            "noise_sigma": noise_sigma,
            "navigation_offset_x": offset_x,
            "navigation_offset_y": offset_y,
            "repetitive_stress_case": bool(is_repetitive_stress_case),
        },
    }


def generate_dataset(out_dir: str, n: int = 30, seed: int = 42, repetitive_stress_prob: float = 0.15):
    os.makedirs(os.path.join(out_dir, "reference"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)

    base_img, motif_centers = _make_synthetic_wafer(seed=seed)
    cv2.imwrite(os.path.join(out_dir, "reference", "base_wafer.png"), base_img)

    rng = np.random.default_rng(seed)
    manifest = []
    for i in range(n):
        sample = generate_sample(base_img, motif_centers, i, rng,
                                  repetitive_stress_prob=repetitive_stress_prob)
        ref_path = os.path.join(out_dir, "reference", f"reference_{i:03d}.png")
        search_path = os.path.join(out_dir, "test", f"test_{i:03d}.png")
        gt_path = os.path.join(out_dir, "test", f"ground_truth_{i:03d}.json")

        cv2.imwrite(ref_path, sample["reference"])
        cv2.imwrite(search_path, sample["search"])
        with open(gt_path, "w") as f:
            json.dump({"ground_truth": sample["ground_truth"], "params": sample["params"]}, f, indent=2)

        manifest.append({
            "sample_id": i, "reference": ref_path, "search": search_path, "ground_truth_file": gt_path,
        })

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {n} samples in {out_dir}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stress_prob", type=float, default=0.15,
                         help="fraction of samples deliberately anchored on the repetitive "
                              "grid pattern (intentional hard cases, Section 7.2)")
    args = parser.parse_args()
    generate_dataset(args.out, n=args.n, seed=args.seed, repetitive_stress_prob=args.stress_prob)
