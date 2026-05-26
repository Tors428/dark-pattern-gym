"""
A dead-simple keyword baseline. It scores each option by how much its label
looks like "a step toward cancelling" versus "a step toward staying," then
picks the best. It needs no API key and runs instantly, so it's useful for:
  - testing that the environment works end to end, and
  - serving as a non-LLM baseline to compare the Gemini agent against.

It is intentionally naive. Watching *where* it gets fooled by the dark UI
(e.g. confirmshaming labels that contain positive-sounding words) is half
the point of the demo.
"""

from typing import List

# Words that suggest progressing toward the goal (cancelling).
GO = ("cancel", "continue", "confirm", "manage", "yes", "proceed", "account",
      "settings", "subscription", "membership")
# Words that suggest the manipulative "stay" path.
STOP = ("keep", "enjoy", "upgrade", "discount", "offer", "save", "benefit", "premium", "no,")


class KeywordAgent:
    name = "keyword-baseline"

    def act(self, observation: str, options: List[str]) -> int:
        best_idx, best_score = 0, float("-inf")
        for i, label in enumerate(options):
            low = label.lower()
            score = 0
            for w in GO:
                if w in low:
                    score += 1
            for w in STOP:
                if w in low:
                    score -= 1
            # Mild tie-breaker: cancellation links tend to be placed last/small,
            # so nudge toward later options when scores are equal.
            score += i * 0.01
            if score > best_score:
                best_idx, best_score = i, score
        return best_idx
