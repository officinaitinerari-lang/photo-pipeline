from src.ai_analyzer import (
    ETSY_TAG_COUNT,
    AIResponseRaw,
    EtsyTextRaw,
    format_gps_subtitle,
    normalise_tags,
    postprocess,
    truncate_words,
)
from src.image_loader import ExifInfo


def _raw(**over):
    data = dict(
        subject_category="architecture",
        location_country="Japan",
        location_place="Kyoto",
        has_recognizable_people=False,
        has_license_plates=False,
        has_logos_or_trademarks=False,
        has_protected_property_or_artwork=False,
        microstock_commercial_safe=True,
        safety_notes="",
        etsy_en=EtsyTextRaw(
            title="Kyoto Temple in Snow Print, Japan Winter Wall Art, Minimalist Japanese Photography Digital Download "
            "Large Printable Poster for Living Room Decor",
            description="A quiet winter morning at a Kyoto temple, snow resting on the curved roofs. " * 3,
            tags=["kyoto print", "japan wall art", "Japanese temple photography print", "snow", "snow", "winter decor"],
        ),
        etsy_it=EtsyTextRaw(
            title="Stampa Tempio di Kyoto sotto la neve",
            description="Una mattina d'inverno silenziosa in un tempio di Kyoto, con la neve sui tetti. " * 2,
            tags=["stampa kyoto", "poster giappone"],
        ),
        poster_title="Kyoto",
        poster_subtitle="JAPAN • WINTER LIGHT",
        microstock_title="Traditional Japanese temple covered in fresh snow in Kyoto, Japan, during a calm winter morning",
        microstock_description="Snow covered wooden temple roofs in Kyoto on a winter morning.",
        microstock_keywords=[f"keyword{i}" for i in range(42)] + ["Keyword1", "kyoto, japan"],
        microstock_category="Buildings and Architecture",
    )
    data.update(over)
    return AIResponseRaw(**data)


def test_tags_exactly_13_and_short():
    res = postprocess(_raw(), ExifInfo())
    for lang in (res.etsy_en, res.etsy_it):
        assert len(lang.tags) == ETSY_TAG_COUNT
        assert len(set(lang.tags)) == ETSY_TAG_COUNT
        assert all(0 < len(t) <= 20 for t in lang.tags)


def test_title_and_keyword_limits():
    res = postprocess(_raw(), ExifInfo())
    assert len(res.etsy_en.title) <= 140
    assert 35 <= len(res.microstock_keywords) <= 45
    assert len(res.microstock_keywords) == len(set(res.microstock_keywords))
    assert "kyoto" in res.microstock_keywords and "japan" in res.microstock_keywords


def test_gps_subtitle_overrides_model_text():
    res = postprocess(_raw(), ExifInfo(latitude=35.0116, longitude=135.7681))
    assert res.poster_subtitle == "JAPAN • 35.0116° N, 135.7681° E"
    assert format_gps_subtitle("Chile", -33.45, -70.66) == "CHILE • 33.4500° S, 70.6600° W"


def test_any_safety_flag_blocks_commercial_use():
    res = postprocess(_raw(has_logos_or_trademarks=True, microstock_commercial_safe=True), ExifInfo())
    assert res.microstock_commercial_safe is False
    assert "logos_trademarks" in res.safety_flags
    assert res.safety_notes


def test_unknown_category_is_mapped():
    res = postprocess(_raw(microstock_category="architecture"), ExifInfo())
    assert res.microstock_category == "Buildings and Architecture"


def test_helpers():
    assert truncate_words("one two three four", 9) == "one two"
    assert normalise_tags(["#Kyoto", "kyoto"], ["a"], count=2) == ["kyoto", "a"]
