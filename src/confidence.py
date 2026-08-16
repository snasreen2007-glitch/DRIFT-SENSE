"""
confidence.py
===============
Implements Document Section 3.8 (Confidence-Aware Localization):

    C = f(S_coarse, S_fine, E)

We combine three signals into a single normalized [0,1] confidence:

  1. S_coarse : the CNN coarse cosine-similarity score of the winning candidate
  2. S_fine   : the NCC score from fine registration
  3. margin   : how much better the winning candidate is than the
                RUNNER-UP candidate (a large margin means the match was
                unambiguous; a small margin is a red flag for repetitive
                wafer patterns -- Section 3.5 / 7.2)

A low confidence should trigger the "pass to secondary verification"
behavior the paper describes.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ConfidenceResult:
    confidence: float
    reliable: bool
    reason: str


def compute_confidence(best_coarse_score: float,
                        best_fine_ncc: float,
                        second_best_fine_ncc: float,
                        reliability_threshold: float = 0.55,
                        min_ncc_floor: float = 0.90) -> ConfidenceResult:
    """min_ncc_floor: an ABSOLUTE minimum fine-registration NCC score
    required before a match can ever be called reliable, regardless of
    how much better it is than the runner-up candidate. This matters
    because on repetitive/textured backgrounds (e.g. this project's grid
    pattern), even a WRONG location can score decently on NCC just from
    generic background self-similarity, while still beating other wrong
    candidates by a comfortable margin. A relative-margin-only check
    can't tell "confidently right" apart from "confidently the best of
    several wrong options" -- the absolute floor catches that case."""
    coarse_term = np.clip((best_coarse_score + 1) / 2, 0, 1)   # cosine sim in [-1,1] -> [0,1]
    fine_term = np.clip((best_fine_ncc + 1) / 2, 0, 1)          # NCC in [-1,1] -> [0,1]
    margin = best_fine_ncc - second_best_fine_ncc
    margin_term = np.clip(margin / 0.25, 0, 1)                  # saturate once margin >= 0.25

    confidence = float(0.3 * coarse_term + 0.5 * fine_term + 0.2 * margin_term)

    below_floor = best_fine_ncc < min_ncc_floor
    if below_floor:
        # hard cap: no amount of margin can rescue a weak absolute match
        confidence = min(confidence, 0.5)

    reliable = (confidence >= reliability_threshold) and not below_floor
    if not reliable:
        if below_floor:
            reason = (f"fine-registration score ({best_fine_ncc:.2f}) below the reliability "
                      f"floor ({min_ncc_floor}) -- target may not actually be present in the "
                      f"search image")
        elif margin_term < 0.3:
            reason = "low separation from runner-up candidate (possible repetitive pattern)"
        elif fine_term < 0.5:
            reason = "weak fine-registration match (possible blur/noise/occlusion)"
        else:
            reason = "weak coarse CNN similarity (possible large appearance change)"
    else:
        reason = "match is unambiguous and well-registered"

    return ConfidenceResult(confidence=confidence, reliable=reliable, reason=reason)


if __name__ == "__main__":
    r = compute_confidence(best_coarse_score=0.9, best_fine_ncc=0.95, second_best_fine_ncc=0.4)
    print(r)
    r2 = compute_confidence(best_coarse_score=0.6, best_fine_ncc=0.55, second_best_fine_ncc=0.53)
    print(r2)
