"""
coarse_matching.py
====================
Implements Document Section 3.4 (Coarse Feature Matching) and
Section 3.5 (Top-K Candidate Selection).

    S(F_R, F_S) = <F_R, F_S> / (||F_R|| ||F_S||)      -- cosine similarity

For every pyramid level we slide the reference embedding over the
dense search feature map (a 1x1 correlation, i.e. a dot product at
every spatial location since both are already L2-normalized) to get a
similarity map. We then keep the top-K peaks across ALL pyramid levels
combined -- this is what makes DRIFT-SENSE robust to repetitive wafer
patterns: instead of committing to a single best match too early, we
carry several plausible candidates into fine registration.
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class Candidate:
    x: float            # x location in ORIGINAL search-image pixel coords
    y: float            # y location in ORIGINAL search-image pixel coords
    scale: float         # pyramid scale factor this candidate came from
    score: float          # coarse cosine-similarity score


def similarity_map(ref_embedding: torch.Tensor, search_fmap: torch.Tensor) -> torch.Tensor:
    """ref_embedding: (C,)   search_fmap: (C,H,W)  ->  (H,W) cosine similarity map."""
    fmap_norm = F.normalize(search_fmap, dim=0)
    sim = torch.einsum("c,chw->hw", ref_embedding, fmap_norm)
    return sim


def topk_candidates(ref_embedding: torch.Tensor,
                     pyramid_fmaps: dict,
                     stride: int,
                     k: int = 5,
                     min_separation_px: int = 12,
                     border_exclude: int = 2) -> list:
    """pyramid_fmaps: {scale: feature_map (C,H',W')} for every pyramid level.

    border_exclude: number of feature-map cells to exclude around the
    edge of each level before taking the argmax. Convolutional feature
    maps (especially from small / undertrained CNNs) tend to have
    inflated, unreliable activations right at the border due to zero
    or replicate padding -- excluding a small ring avoids the detector
    latching onto image edges/corners instead of real content.

    Returns up to k Candidate objects, sorted by descending score, with a
    simple non-max suppression so we don't return k copies of the same peak.
    """
    all_points = []  # (score, x, y, scale)

    for scale, fmap in pyramid_fmaps.items():
        sim = similarity_map(ref_embedding, fmap)              # (H',W') in feature-map coords
        sim_np = sim.detach().cpu().numpy()

        h, w = sim_np.shape
        b = border_exclude
        if h > 2 * b + 1 and w > 2 * b + 1:
            mask = np.full_like(sim_np, -np.inf)
            mask[b:h - b, b:w - b] = sim_np[b:h - b, b:w - b]
            sim_np = mask

        # take a generous number of local peaks per level, NMS narrows later
        flat_idx = np.argsort(sim_np.ravel())[::-1][: k * 4]
        for idx in flat_idx:
            fy, fx = np.unravel_index(idx, sim_np.shape)
            if not np.isfinite(sim_np[fy, fx]):
                continue
            # feature-map coords -> pyramid-level pixel coords -> ORIGINAL image coords
            px = (fx * stride) / scale
            py = (fy * stride) / scale
            all_points.append((float(sim_np[fy, fx]), px, py, scale))

    all_points.sort(key=lambda t: t[0], reverse=True)

    kept = []
    for score, x, y, scale in all_points:
        if all(((x - c.x) ** 2 + (y - c.y) ** 2) ** 0.5 > min_separation_px for c in kept):
            kept.append(Candidate(x=x, y=y, scale=scale, score=score))
        if len(kept) >= k:
            break

    return kept


if __name__ == "__main__":
    torch.manual_seed(0)
    ref_emb = F.normalize(torch.rand(128), dim=0)
    fmap1 = torch.rand(128, 40, 40)
    fmap2 = torch.rand(128, 28, 28)
    cands = topk_candidates(ref_emb, {1.0: fmap1, 0.7: fmap2}, stride=8, k=5)
    for c in cands:
        print(c)
