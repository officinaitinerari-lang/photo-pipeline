"""Gemini Vision analysis: bilingual Etsy SEO, microstock metadata and commercial-safety check."""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from .image_loader import ExifInfo

log = logging.getLogger(__name__)

ADOBE_STOCK_CATEGORIES = [
    "Animals", "Buildings and Architecture", "Business", "Drinks", "The Environment",
    "States of Mind", "Food", "Graphic Resources", "Hobbies and Leisure", "Industry",
    "Landscapes", "Lifestyle", "People", "Plants and Flowers", "Culture and Religion",
    "Science", "Social Issues", "Sports", "Technology", "Transport", "Travel",
]

ETSY_TITLE_MAX = 140
ETSY_TAG_COUNT = 13
ETSY_TAG_MAX = 20
MICRO_TITLE_MAX = 200
MICRO_KW_MIN = 35
MICRO_KW_MAX = 45


class AIQuotaExhaustedError(Exception):
    """Free-tier quota exhausted after all retries: the job should be deferred, not failed."""


class AIAnalysisError(Exception):
    """Non-recoverable AI failure (bad key, invalid output after retries...)."""


# ------------------------------------------------------------------ Gemini-facing schema
class EtsyTextRaw(BaseModel):
    title: str
    description: str
    tags: list[str]


class AIResponseRaw(BaseModel):
    """Schema sent to Gemini (kept constraint-free: limits are enforced in post-processing)."""

    subject_category: str = Field(description="landscape | architecture | street | minimal | other")
    location_country: str = Field(description="Country in English, empty if unknown")
    location_place: str = Field(description="City/place/area in English, empty if unknown")
    has_recognizable_people: bool
    has_license_plates: bool
    has_logos_or_trademarks: bool
    has_protected_property_or_artwork: bool
    microstock_commercial_safe: bool
    safety_notes: str
    etsy_en: EtsyTextRaw
    etsy_it: EtsyTextRaw
    poster_title: str
    poster_subtitle: str
    microstock_title: str
    microstock_description: str
    microstock_keywords: list[str]
    microstock_category: str


# ------------------------------------------------------------------ validated result
class EtsyText(BaseModel):
    title: str = Field(max_length=ETSY_TITLE_MAX)
    description: str = Field(min_length=40)
    tags: list[str] = Field(min_length=ETSY_TAG_COUNT, max_length=ETSY_TAG_COUNT)

    @field_validator("tags")
    @classmethod
    def _tag_len(cls, v: list[str]) -> list[str]:
        for t in v:
            if not 0 < len(t) <= ETSY_TAG_MAX:
                raise ValueError(f"tag {t!r} violates 1..{ETSY_TAG_MAX} chars")
        return v


class AIAnalysis(BaseModel):
    subject_category: str
    location_country: str
    location_place: str
    microstock_commercial_safe: bool
    safety_notes: str
    safety_flags: list[str]
    etsy_en: EtsyText
    etsy_it: EtsyText
    poster_title: str = Field(min_length=1, max_length=40)
    poster_subtitle: str = Field(max_length=90)
    microstock_title: str = Field(min_length=10, max_length=MICRO_TITLE_MAX)
    microstock_description: str = Field(min_length=10)
    microstock_keywords: list[str] = Field(min_length=MICRO_KW_MIN, max_length=MICRO_KW_MAX)
    microstock_category: str


# ------------------------------------------------------------------ normalisation helpers
def truncate_words(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:-|•")


def _clean_token(t: str) -> str:
    t = re.sub(r"[#\"'`]", "", t or "")
    return re.sub(r"\s+", " ", t).strip(" ,.;:").lower()


def normalise_tags(tags: list[str], fallbacks: list[str], count: int = ETSY_TAG_COUNT, max_len: int = ETSY_TAG_MAX) -> list[str]:
    """Exactly `count` unique tags, each <= max_len chars (truncating at word boundaries)."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in [*tags, *fallbacks]:
        t = truncate_words(_clean_token(raw), max_len)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) == count:
            break
    return out


def normalise_keywords(keywords: list[str], max_count: int = MICRO_KW_MAX) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in keywords:
        for piece in str(raw).split(","):
            k = _clean_token(piece)
            if k and len(k) <= 40 and k not in seen:
                seen.add(k)
                out.append(k)
    return out[:max_count]


def format_gps_subtitle(country: str, lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    coords = f"{abs(lat):.4f}° {ns}, {abs(lon):.4f}° {ew}"
    return f"{country.upper()} • {coords}" if country else coords


def postprocess(raw: AIResponseRaw, exif: ExifInfo) -> AIAnalysis:
    flags = [
        name for name, on in (
            ("recognizable_people", raw.has_recognizable_people),
            ("license_plates", raw.has_license_plates),
            ("logos_trademarks", raw.has_logos_or_trademarks),
            ("protected_property", raw.has_protected_property_or_artwork),
        ) if on
    ]
    safe = bool(raw.microstock_commercial_safe) and not flags
    notes = raw.safety_notes.strip()
    if not safe and not notes:
        notes = "Potential release/IP issue: " + (", ".join(flags) or "flagged by model")

    keywords = normalise_keywords(raw.microstock_keywords)
    fallback_tags = [*keywords, raw.location_place, raw.location_country, raw.subject_category, "wall art", "digital print"]

    subtitle = raw.poster_subtitle.strip()
    if exif.has_gps and exif.latitude is not None and exif.longitude is not None:
        subtitle = format_gps_subtitle(raw.location_country, exif.latitude, exif.longitude)

    category = raw.microstock_category.strip()
    if category not in ADOBE_STOCK_CATEGORIES:
        match = next((c for c in ADOBE_STOCK_CATEGORIES if c.lower() in category.lower() or category.lower() in c.lower()), None)
        category = match or ("Landscapes" if raw.subject_category == "landscape" else "Travel")

    return AIAnalysis(
        subject_category=raw.subject_category.strip().lower(),
        location_country=raw.location_country.strip(),
        location_place=raw.location_place.strip(),
        microstock_commercial_safe=safe,
        safety_notes=notes,
        safety_flags=flags,
        etsy_en=EtsyText(
            title=truncate_words(raw.etsy_en.title, ETSY_TITLE_MAX),
            description=raw.etsy_en.description.strip(),
            tags=normalise_tags(raw.etsy_en.tags, fallback_tags),
        ),
        etsy_it=EtsyText(
            title=truncate_words(raw.etsy_it.title, ETSY_TITLE_MAX),
            description=raw.etsy_it.description.strip(),
            tags=normalise_tags(raw.etsy_it.tags, [*raw.etsy_en.tags, *fallback_tags]),
        ),
        poster_title=truncate_words(raw.poster_title.upper(), 40) or (raw.location_place or "UNTITLED").upper()[:40],
        poster_subtitle=truncate_words(subtitle, 90),
        microstock_title=truncate_words(raw.microstock_title, MICRO_TITLE_MAX),
        microstock_description=truncate_words(raw.microstock_description, 1900),
        microstock_keywords=keywords,
        microstock_category=category,
    )


# ------------------------------------------------------------------ prompt
def build_prompt(context_text: str, exif: ExifInfo, is_analog: bool, orientation: str) -> str:
    hints = {
        "folder_and_filename_context": context_text or "(none)",
        "exif": exif.as_prompt_dict() or "(none)",
        "analog_film_scan": is_analog,
        "orientation": orientation,
    }
    return f"""You are an expert fine-art print merchandiser, Etsy SEO copywriter and microstock reviewer.
Analyse the attached photograph. Subjects are typically natural landscapes, architecture/cities,
street photography or minimalist details.

Context hints (may be incomplete or absent; if missing, infer from the image itself, never invent
precise facts you cannot see or deduce):
{json.dumps(hints, ensure_ascii=False, indent=2)}

LEGAL CHECK for commercial microstock licensing (be strict):
- has_recognizable_people: any human face/person identifiable without a model release.
- has_license_plates: readable vehicle plates.
- has_logos_or_trademarks: readable brand names, logos, shop signs with trademarks.
- has_protected_property_or_artwork: copyrighted art, murals, sculptures, famous buildings whose
  image rights are restricted (e.g. Eiffel Tower lit at night), private property interiors.
- microstock_commercial_safe = true ONLY if none of the above apply. safety_notes: short reason.

ETSY (digital download wall-art print), produce English (etsy_en) and Italian (etsy_it):
- title: SEO-rich, natural, max 140 characters, include "Print"/"Wall Art"/"Digital Download"
  style terms (Italian: "Stampa", "Poster", "Download Digitale").
- description: 2 evocative paragraphs about atmosphere and place, then why it suits home decor.
  Do NOT list file sizes (they are appended automatically).
- tags: exactly 13 tags, each max 20 characters, no duplicates, buyer search phrases.
- poster_title: 1-3 words, UPPERCASE, the place or subject (e.g. "KYOTO").
- poster_subtitle: e.g. "JAPAN • 35.0116° N, 135.7681° E" when GPS is known, otherwise a short
  geographic or poetic line in English uppercase (max 60 chars).
{"- This is an ANALOG FILM scan: mention film grain/analog character in descriptions." if is_analog else ""}

MICROSTOCK (English only):
- microstock_title: descriptive commercial title, max 200 characters, no keyword stuffing.
- microstock_description: objective factual description (what, where, conditions).
- microstock_keywords: 40 single words or short 2-word phrases ordered by relevance, the first 10
  being the most important, no duplicates, no trademarks.
- microstock_category: one of {", ".join(ADOBE_STOCK_CATEGORIES)}.
- location_country / location_place in English (empty string if unknown).
Return only JSON matching the schema."""


# ------------------------------------------------------------------ rate limiter
class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


class GeminiAnalyzer:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        min_interval: float = 6.0,
        max_retries: int = 6,
        backoff_base: float = 4.0,
        backoff_max: float = 120.0,
        timeout_seconds: int = 120,
        thinking_budget: int = -1,
    ) -> None:
        from google import genai
        from google.genai import types

        if not api_key:
            raise AIAnalysisError("GEMINI_API_KEY is not configured")
        self._types = types
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_seconds * 1000))
        self.model = model
        self.limiter = RateLimiter(min_interval)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.thinking_budget = thinking_budget

    def _config(self) -> Any:
        types = self._types
        kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": AIResponseRaw,
            "temperature": 0.5,
        }
        if self.thinking_budget >= 0:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return types.GenerateContentConfig(**kwargs)

    def _call(self, jpeg: bytes, prompt: str) -> AIResponseRaw:
        from google.genai import errors

        last_exc: Exception | None = None
        saw_quota = False
        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait()
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=[self._types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
                    config=self._config(),
                )
                parsed = resp.parsed
                if isinstance(parsed, AIResponseRaw):
                    return parsed
                if not resp.text:
                    raise AIAnalysisError("Empty response (possibly blocked by safety filters)")
                return AIResponseRaw.model_validate_json(resp.text)
            except errors.APIError as exc:
                code = int(getattr(exc, "code", 0) or 0)
                last_exc = exc
                if code in (400, 401, 403, 404):
                    raise AIAnalysisError(f"Gemini API error {code}: {exc}") from exc
                if code not in _RETRYABLE_CODES:
                    raise AIAnalysisError(f"Gemini API error {code}: {exc}") from exc
                saw_quota = saw_quota or code == 429
            except (ValidationError, json.JSONDecodeError) as exc:
                last_exc = exc
                log.warning("Gemini returned invalid JSON", extra={"attempt": attempt, "error": str(exc)[:300]})
            except AIAnalysisError:
                raise
            except Exception as exc:
                last_exc = exc
            delay = min(self.backoff_max, self.backoff_base * 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
            log.warning(
                "Gemini call failed, backing off",
                extra={"attempt": attempt, "delay_s": round(delay, 1), "error": str(last_exc)[:300]},
            )
            if attempt < self.max_retries:
                time.sleep(delay)
        if saw_quota:
            raise AIQuotaExhaustedError(f"Gemini quota/rate limit persisted: {last_exc}")
        raise AIAnalysisError(f"Gemini failed after {self.max_retries} attempts: {last_exc}")

    def analyse(
        self, jpeg_preview: bytes, context_text: str, exif: ExifInfo, is_analog: bool, orientation: str
    ) -> AIAnalysis:
        prompt = build_prompt(context_text, exif, is_analog, orientation)
        last_error: Exception | None = None
        for validation_round in range(2):
            raw = self._call(jpeg_preview, prompt)
            try:
                result = postprocess(raw, exif)
                log.info(
                    "AI analysis ok",
                    extra={
                        "poster_title": result.poster_title,
                        "commercial_safe": result.microstock_commercial_safe,
                        "keywords": len(result.microstock_keywords),
                    },
                )
                return result
            except ValidationError as exc:
                last_error = exc
                log.warning("AI output failed validation", extra={"round": validation_round, "error": str(exc)[:500]})
                prompt += (
                    "\n\nIMPORTANT: your previous answer was rejected: "
                    f"{str(exc)[:400]}. Provide at least {MICRO_KW_MIN + 3} distinct microstock keywords "
                    "and exactly 13 short Etsy tags per language."
                )
        raise AIAnalysisError(f"AI output invalid after retries: {last_error}")
