from __future__ import annotations

from typing import Iterable

SYSTEM_PROMPT = """You are a helpful assistant that can analyze multimodal content,
follow a Chain-of-Thought process, and provide comprehensive feature analysis
in a specific JSON format.

Analyze only the modalities explicitly provided. Do not infer unavailable raw
content as if it had been observed. Return a valid JSON object in <answer>.
"""

MODALITY_LABELS = {"text": "text", "vision": "image/video", "audio": "audio"}


def build_user_prompt(
    available: Iterable[str],
    missing: Iterable[str],
    feedback: str | None = None,
) -> str:
    available = list(available)
    missing = list(missing)
    observed = ", ".join(MODALITY_LABELS[name] for name in available)
    targets = ", ".join(MODALITY_LABELS[name] for name in missing)
    feedback_block = ""
    if feedback:
        feedback_block = f"\nVerifier feedback from the previous round:\n{feedback}\n"
    return f"""Observed modalities: {observed}.
Missing targets: {targets or "none; analyze the complete input"}.
{feedback_block}
Follow these steps:
1. Identify entities supported by the observed modalities.
2. Describe actions and affective evidence without inventing missing content.
3. Infer constraints useful for reconstructing the missing target modalities.
4. Return the final plan using exactly this schema:

<answer>
{{
  "entity": "short entity description",
  "action": "short action or communicative-intent description",
  "attribute": {{
    "valence": 0.0,
    "arousal": 0.5,
    "dominance": 0.5,
    "intensity": "weak"
  }}
}}
</answer>

Valence must be in [-1,1], arousal/dominance in [0,1], and intensity one of
"weak", "moderate", or "strong". If evidence is ambiguous, use
valence=0, arousal=0.5, dominance=0.5, intensity="weak".
"""


def diagnostic_feedback(field: str, content: str, score: float) -> str:
    return (
        f"The {field} constraint {content!r} has low consistency "
        f"(score: {score:.4f}). Revise this field using only available observations."
    )

