from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"


@dataclass(frozen=True)
class GeminiSelection:
    chosen_rank: int
    reason: str
    model: str
    raw_text: str
    used_fallback: bool


def _get_gemini_api_key() -> str | None:
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return None


def _build_prompt(company: str, places: list[dict[str, Any]]) -> str:
    lines = [
        f"{place['rank']}. lat={place['lat']:.6f}, lon={place['lon']:.6f} (score={place['score']:.4f})"
        for place in places
    ]
    coordinate_block = "\n".join(lines)

    return (
        f"Here are 10 coordinates of places. Pick the one that fits the vibe of {company} best.\n\n"
        "Return strict JSON with this schema: "
        '{"chosen_rank": <integer 1-10>, "reason": <short string>} and nothing else.\n\n'
        f"Coordinates:\n{coordinate_block}\n"
    )


def _extract_text_from_response(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not candidates:
        return ""

    first = candidates[0]
    content = first.get("content", {})
    parts = content.get("parts", [])
    if not parts:
        return ""

    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "\n".join([txt for txt in texts if txt]).strip()


def _parse_rank_and_reason(raw_text: str, max_rank: int) -> tuple[int, str]:
    stripped = raw_text.strip()

    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    try:
        data = json.loads(stripped)
        rank = int(data.get("chosen_rank", 1))
        reason = str(data.get("reason", "No reason provided by Gemini.")).strip()
        if 1 <= rank <= max_rank:
            return rank, reason or "No reason provided by Gemini."
    except Exception:
        pass

    number_match = re.search(r"\b(10|[1-9])\b", raw_text)
    if number_match:
        rank = int(number_match.group(1))
        if 1 <= rank <= max_rank:
            return rank, "Parsed from non-JSON Gemini response."

    return 1, "Could not parse Gemini output; defaulted to first candidate."


def choose_coordinate(
    company: str,
    places: list[dict[str, Any]],
    model: str = DEFAULT_GEMINI_MODEL,
    timeout: int = 30,
) -> GeminiSelection:
    if not places:
        return GeminiSelection(
            chosen_rank=1,
            reason="No candidates available.",
            model=model,
            raw_text="",
            used_fallback=True,
        )

    api_key = _get_gemini_api_key()
    if not api_key:
        return GeminiSelection(
            chosen_rank=1,
            reason="GEMINI_API_KEY is missing; defaulted to first candidate.",
            model=model,
            raw_text="",
            used_fallback=True,
        )

    prompt = _build_prompt(company=company, places=places)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 256,
        },
    }

    url = f"{GEMINI_API_BASE}/{model}:generateContent"
    try:
        response = requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        raw_text = _extract_text_from_response(data)
        rank, reason = _parse_rank_and_reason(raw_text, max_rank=len(places))
        return GeminiSelection(
            chosen_rank=rank,
            reason=reason,
            model=model,
            raw_text=raw_text,
            used_fallback=False,
        )
    except Exception as exc:
        return GeminiSelection(
            chosen_rank=1,
            reason=f"Gemini request failed: {exc}. Defaulted to first candidate.",
            model=model,
            raw_text="",
            used_fallback=True,
        )
