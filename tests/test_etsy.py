import json
from pathlib import Path

import httpx
import pytest

from src.config import Settings
from src.etsy_publisher import (
    MARKER,
    EtsyClient,
    EtsyError,
    clean_tags,
    clean_title,
    parse_listing_file,
    when_made_candidates,
)

LISTING = """ETSY LISTING — slug
Gallery Poster: X

========================================================================
ENGLISH
========================================================================

TITLE / TITOLO (20/140):
English title: one & two & three

DESCRIPTION / DESCRIZIONE:
English body.

— WHAT YOU GET —
• stuff

TAGS (3):
italy print, wall art, tivoli

========================================================================
ITALIANO
========================================================================

TITLE / TITOLO (20/140):
Stampa Tivoli Italia, Poster

DESCRIPTION / DESCRIZIONE:
Corpo italiano.

— COSA RICEVI —
• roba

TAGS (4):
stampa tivoli, quadro_roma, una etichetta troppo lunga davvero, stampa tivoli
"""


def _settings(tmp_path: Path, **kw) -> Settings:
    return Settings(data_root=tmp_path, etsy_enabled=True, etsy_api_key="KEY", etsy_shared_secret="SEC",
                    etsy_redirect_uri="https://localhost/etsy-callback", dry_run=False, **kw)


def _package(tmp_path: Path, n_zips: int = 2) -> Path:
    d = tmp_path / "output" / "etsy" / "tivoli-slug"
    (d / "mockups").mkdir(parents=True)
    (d / "etsy_listing.txt").write_text(LISTING, encoding="utf-8")
    for i in range(1, 4):
        (d / "mockups" / f"mockup_0{i}.jpg").write_bytes(b"jpg")
    for i in range(1, n_zips + 1):
        (d / f"prints_fullbleed_part{i}.zip").write_bytes(b"zip")
    return d


def test_parse_italian_section(tmp_path):
    p = tmp_path / "etsy_listing.txt"
    p.write_text(LISTING, encoding="utf-8")
    it = parse_listing_file(p, "it")
    assert it.title == "Stampa Tivoli Italia, Poster"
    assert it.description.startswith("Corpo italiano.") and "COSA RICEVI" in it.description
    assert it.tags == ["stampa tivoli", "quadro roma"]
    en = parse_listing_file(p, "en")
    assert en.title == "English title: one & two three"


def test_clean_helpers():
    assert clean_title("a+b+c%d%") == "a+bc%d"
    assert len(clean_title("parola " * 40)) <= 140
    assert clean_tags(["ok tag", "bad!tag", "x" * 21] + [f"t{i}" for i in range(20)])[:2] == ["ok tag", "bad tag"]
    assert len(clean_tags([f"t{i}" for i in range(20)])) == 13


def test_when_made_candidates():
    import datetime as dt

    assert when_made_candidates("", dt.date(2026, 11, 26))[0] == "2020_2026"
    assert when_made_candidates("2020_2023") == ["2020_2023"]


def test_file_based_authorisation(tmp_path):
    s = _settings(tmp_path)
    s.state_dir.mkdir(parents=True)
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        assert req.url.path == "/v3/public/oauth/token"
        body = dict(x.split("=", 1) for x in req.content.decode().split("&"))
        assert body["grant_type"] == "authorization_code" and body["code"] == "THECODE"
        return httpx.Response(200, json={"access_token": "1.a", "refresh_token": "1.r", "expires_in": 3600})

    c = EtsyClient(s, httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.ensure_auth() is False
    url = c.url_path.read_text().strip()
    assert url.startswith("https://www.etsy.com/oauth/connect?") and "code_challenge_method=S256" in url
    state = json.loads(c.pending_path.read_text())["state"]
    c.code_path.write_text(f"https://localhost/etsy-callback?code=THECODE&state={state}")
    assert c.ensure_auth() is True
    assert json.loads(c.token_path.read_text())["access_token"] == "1.a"
    assert not c.code_path.exists() and not c.url_path.exists()


def test_state_mismatch_is_rejected(tmp_path):
    s = _settings(tmp_path)
    s.state_dir.mkdir(parents=True)
    c = EtsyClient(s, httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    c.ensure_auth()
    c.code_path.write_text("https://localhost/etsy-callback?code=X&state=wrong")
    with pytest.raises(EtsyError):
        c.ensure_auth()


class FakeEtsy:
    def __init__(self, fail_file_once: bool = False):
        self.requests: list[httpx.Request] = []
        self.fail_file_once = fail_file_once

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        p = req.url.path
        assert req.headers["x-api-key"] == "KEY:SEC"
        if p == "/v3/application/users/me":
            return httpx.Response(200, json={"user_id": 1, "shop_id": 77})
        if p == "/v3/application/seller-taxonomy/nodes":
            return httpx.Response(200, json={"results": [{"id": 66, "name": "Art & Collectibles", "children": [
                {"id": 2078, "name": "Digital Prints", "children": []}]}]})
        if p == "/v3/application/shops/77/listings":
            body = req.content.decode()
            if "when_made=2020_2026" in body:
                return httpx.Response(400, json={"error": "when_made must be one of ..."})
            assert "type=download" in body and "taxonomy_id=2078" in body and "price=4.90" in body
            return httpx.Response(201, json={"listing_id": 555})
        if p.endswith("/images"):
            return httpx.Response(201, json={"listing_image_id": 1})
        if p.endswith("/files"):
            if self.fail_file_once:
                self.fail_file_once = False
                return httpx.Response(500, text="boom")
            return httpx.Response(201, json={"listing_file_id": 1})
        return httpx.Response(404)


def _with_token(s: Settings) -> None:
    s.state_dir.mkdir(parents=True, exist_ok=True)
    (s.state_dir / "etsy_token.json").write_text(json.dumps(
        {"access_token": "1.a", "refresh_token": "1.r", "expires_at": 9e12}))


def test_sync_creates_complete_draft_and_resumes(tmp_path, monkeypatch):
    import src.etsy_publisher as ep

    monkeypatch.setattr(ep, "when_made_candidates", lambda c, today=None: ["2020_2026", "2020_2025"])
    s = _settings(tmp_path)
    _with_token(s)
    d = _package(tmp_path, n_zips=2)
    fake = FakeEtsy(fail_file_once=True)
    c = EtsyClient(s, httpx.Client(transport=httpx.MockTransport(fake)))

    assert c.sync() == 0  # first file upload fails: draft stays incomplete
    marker = json.loads((d / MARKER).read_text())
    assert marker["status"] == "incomplete" and marker["listing_id"] == 555 and len(marker["images"]) == 3

    assert c.sync() == 1  # resumes: no second listing, only the missing files
    marker = json.loads((d / MARKER).read_text())
    assert marker["status"] == "complete" and len(marker["files"]) == 2
    creates = [r for r in fake.requests if r.url.path == "/v3/application/shops/77/listings"]
    assert len(creates) == 2  # one refused era + one success, never a duplicate draft
    assert c.sync() == 0  # nothing left to do


def test_sync_skips_marked_and_oversized(tmp_path):
    s = _settings(tmp_path)
    _with_token(s)
    d = _package(tmp_path, n_zips=6)
    fake = FakeEtsy()
    c = EtsyClient(s, httpx.Client(transport=httpx.MockTransport(fake)))
    assert c.sync() == 0
    assert not any(r.url.path.endswith("/listings") for r in fake.requests)
    (d / MARKER).write_text(json.dumps({"status": "skip"}))
    assert c.sync() == 0


def test_refresh_on_expiry(tmp_path):
    s = _settings(tmp_path)
    s.state_dir.mkdir(parents=True)
    (s.state_dir / "etsy_token.json").write_text(json.dumps(
        {"access_token": "1.old", "refresh_token": "1.r", "expires_at": 0}))

    def handler(req):
        if req.url.path == "/v3/public/oauth/token":
            assert b"grant_type=refresh_token" in req.content
            return httpx.Response(200, json={"access_token": "1.new", "refresh_token": "1.r2", "expires_in": 3600})
        assert req.headers["Authorization"] == "Bearer 1.new"
        return httpx.Response(200, json={"user_id": 1, "shop_id": 9})

    c = EtsyClient(s, httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.shop_id() == 9
    assert json.loads(c.token_path.read_text())["refresh_token"] == "1.r2"


def test_runtime_validation(tmp_path):
    s = Settings(data_root=tmp_path, gemini_api_key="g", etsy_enabled=True)
    assert any("ETSY_API_KEY" in p for p in s.validate_runtime())
