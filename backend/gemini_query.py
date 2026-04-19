from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from google import genai
import requests


@dataclass(frozen=True)
class GeminiSelection:
    chosen_rank: int
    reason: str
    raw_text: str
    used_fallback: bool


def _build_prompt(company: str, places: list[dict[str, Any]]) -> str:
    lines = [
        f"{place['rank']}. lat={place['lat']:.6f}, lon={place['lon']:.6f} (score={place['score']:.4f})"
        for place in places
    ]
    coordinate_block = "\n".join(lines)

    return (
        f"Here are 10 coordinates of places. Pick the one that fits the vibe of {company} best. Ignore the similarity score, take into account company data, what they do, where they are located etc. Write an elaborate explanation for this choice. make it fast as well, you have 30 seconds to process.\n\n"
        "Return strict JSON with this schema: "
        '{"chosen_rank": <integer 1-10>, "reason": <short string>} and nothing else.\n\n'
        f"Coordinates:\n{coordinate_block}\n"
    )


def _parse_rank_and_reason(raw_text: str, max_rank: int) -> tuple[int, str]:
    stripped = raw_text.strip()

    # Remove code fences if present
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    # Try JSON parsing
    try:
        data = json.loads(stripped)
        rank = int(data.get("chosen_rank", 1))
        reason = str(data.get("reason", "No reason provided by Gemini.")).strip()
        if 1 <= rank <= max_rank:
            return rank, reason or "No reason provided by Gemini."
    except Exception:
        pass

    # Fallback: extract first number 1–10
    number_match = re.search(r"\b(10|[1-9])\b", raw_text)
    if number_match:
        rank = int(number_match.group(1))
        if 1 <= rank <= max_rank:
            return rank, "Parsed from non-JSON Gemini response."

    return 1, "Could not parse Gemini output; defaulted to first candidate."


def choose_coordinate(
    company: str,
    places: list[dict[str, Any]],
    timeout: int = 30,
) -> GeminiSelection:

    if not places:
        return GeminiSelection(
            chosen_rank=1,
            reason="No candidates available.",
            raw_text="",
            used_fallback=True,
        )

    prompt = _build_prompt(company=company, places=places)

    try:
        # Create client (your import requires this style)
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

        # Generate content
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )

        # The Gemini client returns an object with `.text`
        raw_text = response.text if hasattr(response, "text") else str(response)

        # Parse rank + reason
        rank, reason = _parse_rank_and_reason(raw_text, max_rank=len(places))

        return GeminiSelection(
            chosen_rank=rank,
            reason=reason,
            raw_text=raw_text,
            used_fallback=False,
        )

    except Exception as exc:
        return GeminiSelection(
            chosen_rank=1,
            reason=f"Gemini request failed: {exc}. Defaulted to first candidate.",
            raw_text="",
            used_fallback=True,
        )
