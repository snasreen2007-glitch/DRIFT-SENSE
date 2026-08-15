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
                        reliability_threshold: float = 0.55) -> ConfidenceResult:
    coarse_term = np.clip((best_coarse_score + 1) / 2, 0, 1)   # cosine sim in [-1,1] -> [0,1]
    fine_term = np.clip((best_fine_ncc + 1) / 2, 0, 1)          # NCC in [-1,1] -> [0,1]
    margin = best_fine_ncc - second_best_fine_ncc
    margin_term = np.clip(margin / 0.25, 0, 1)                  # saturate once margin >= 0.25

    confidence = float(0.3 * coarse_term + 0.5 * fine_term + 0.2 * margin_term)

    reliable = confidence >= reliability_threshold
    if not reliable:
        if margin_term < 0.3:
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
