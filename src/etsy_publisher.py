"""Create Etsy DRAFT listings from the pipeline's Etsy packages (Open API v3).

Drafts are never activated here: the seller reviews each draft and publishes it by hand on Etsy,
which is also when Etsy charges the listing fee.

Authorisation (OAuth 2.0 + PKCE) is file-driven so it works without a terminal in the container:
1. With no token, the pipeline writes the Etsy authorisation link to ``state/etsy_auth_url.txt``.
2. The seller opens it, clicks "Allow"; Etsy redirects to the registered https redirect URI.
3. The full address of that page (or just the ``code`` value) is saved as ``state/etsy_auth_code.txt``.
4. On the next sync the pipeline exchanges it for tokens (``state/etsy_token.json``) and deletes it.
Access tokens last one hour and are refreshed automatically; the refresh token lasts 90 days and is
renewed on every refresh, so a pipeline that syncs regularly never needs a new authorisation.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from .config import Settings

log = logging.getLogger(__name__)

API_BASE = "https://api.etsy.com/v3/application"
TOKEN_URL = "https://api.etsy.com/v3/public/oauth/token"
CONNECT_URL = "https://www.etsy.com/oauth/connect"
SCOPES = "listings_r listings_w shops_r"
MARKER = "etsy_draft.json"
MAX_FILES = 5
MAX_TAGS = 13
MAX_TAG_LEN = 20
MAX_TITLE_LEN = 140

_TAG_BAD = re.compile(r"[^\w\s\-'™©®]|_", re.UNICODE)
_SECTION_TITLES = {"it": "ITALIANO", "en": "ENGLISH"}
_OTHER_LANGUAGE = {"it": "en", "en": "it"}


class EtsyError(Exception):
    pass


class EtsyAuthPending(EtsyError):
    """No usable token yet: the seller still has to authorise the app."""


@dataclass
class ListingText:
    title: str
    description: str
    tags: list[str]


# ----------------------------------------------------------------------------- text helpers
def parse_listing_file(path: Path, language: str) -> ListingText:
    """Read title, description and tags for one language from the pipeline's etsy_listing.txt."""
    text = path.read_text(encoding="utf-8")
    header = _SECTION_TITLES.get(language.lower(), "ENGLISH")
    parts = re.split(r"\n=+\n(ENGLISH|ITALIANO)\n=+\n", text)
    sections = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    if header not in sections:
        raise EtsyError(f"{path}: section {header} not found")
    body = sections[header]
    title_m = re.search(r"TITLE / TITOLO \(\d+/\d+\):\n(.+)", body)
    desc_m = re.search(r"DESCRIPTION / DESCRIZIONE:\n(.*?)\n\nTAGS \(\d+\):", body, re.S)
    tags_m = re.search(r"TAGS \(\d+\):\n(.+)", body)
    if not (title_m and desc_m and tags_m):
        raise EtsyError(f"{path}: cannot parse the {header} section")
    return ListingText(
        title=clean_title(title_m.group(1)),
        description=desc_m.group(1).strip(),
        tags=clean_tags(tags_m.group(1).split(",")),
    )


def clean_title(title: str) -> str:
    """Etsy allows %, :, & and + only once each and at most 140 characters."""
    out: list[str] = []
    seen: set[str] = set()
    for ch in title.strip():
        if ch in "%:&+":
            if ch in seen:
                continue
            seen.add(ch)
        out.append(ch)
    title = re.sub(r"\s{2,}", " ", "".join(out))
    if len(title) > MAX_TITLE_LEN:
        title = title[:MAX_TITLE_LEN].rsplit(" ", 1)[0].rstrip(" ,;-")
    return title


def clean_tags(raw: list[str]) -> list[str]:
    """Letters, digits, spaces, - ' ™ © ® only; max 20 characters each; max 13, no duplicates."""
    tags: list[str] = []
    for t in raw:
        t = re.sub(r"\s+", " ", _TAG_BAD.sub(" ", t)).strip()
        if not t or len(t) > MAX_TAG_LEN or t.lower() in {x.lower() for x in tags}:
            continue
        tags.append(t)
        if len(tags) == MAX_TAGS:
            break
    return tags


def package_contents(slug_dir: Path) -> tuple[list[Path], list[Path]]:
    zips = sorted(slug_dir.glob("*.zip"), key=lambda p: (not p.name.startswith("prints_fullbleed"), p.name))
    mockups = sorted((slug_dir / "mockups").glob("*.jpg"))
    return zips, mockups


def when_made_candidates(configured: str, today: _dt.date | None = None) -> list[str]:
    """Etsy's era enum ends with the current year ('2020_2026'); try it, then the previous ones."""
    if configured:
        return [configured]
    year = (today or _dt.date.today()).year
    return [f"2020_{y}" for y in range(year, year - 4, -1)]


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


# ----------------------------------------------------------------------------- client
class EtsyClient:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self.s = settings
        self.http = http or httpx.Client(timeout=120)
        state = settings.state_dir
        self.token_path = state / "etsy_token.json"
        self.pending_path = state / "etsy_oauth_pending.json"
        self.url_path = state / "etsy_auth_url.txt"
        self.code_path = state / "etsy_auth_code.txt"
        self.cache_path = state / "etsy_cache.json"

    # ------------------------------------------------------------------ auth
    @property
    def _api_key_header(self) -> str:
        return f"{self.s.etsy_api_key}:{self.s.etsy_shared_secret}"

    def ensure_auth(self) -> bool:
        """True when a token is available; otherwise drive the file-based authorisation."""
        if self.token_path.exists():
            return True
        if self.code_path.exists() and self.pending_path.exists():
            raw = self.code_path.read_text(encoding="utf-8").strip()
            try:
                self.complete_authorization(raw)
            finally:
                self.code_path.unlink(missing_ok=True)
            return True
        if not self.url_path.exists() or not self.pending_path.exists():
            self.start_authorization()
        return False

    def start_authorization(self) -> str:
        verifier = secrets.token_urlsafe(64)[:96]
        state = secrets.token_urlsafe(16)
        challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
        url = CONNECT_URL + "?" + urlencode({
            "response_type": "code",
            "client_id": self.s.etsy_api_key,
            "redirect_uri": self.s.etsy_redirect_uri,
            "scope": SCOPES,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        })
        self.s.state_dir.mkdir(parents=True, exist_ok=True)
        self.pending_path.write_text(json.dumps({"verifier": verifier, "state": state}), encoding="utf-8")
        self.url_path.write_text(url + "\n", encoding="utf-8")
        log.warning("Etsy authorisation needed: open the link in state/etsy_auth_url.txt")
        return url

    def complete_authorization(self, raw: str) -> None:
        pending = json.loads(self.pending_path.read_text(encoding="utf-8"))
        code = raw
        if raw.startswith("http"):
            q = parse_qs(urlparse(raw).query)
            if "error" in q:
                raise EtsyError(f"Etsy authorisation refused: {q.get('error_description', q['error'])[0]}")
            if q.get("state", [None])[0] != pending["state"]:
                raise EtsyError("Etsy authorisation: state mismatch, start again")
            code = q.get("code", [""])[0]
        if not code:
            raise EtsyError("Etsy authorisation: no code found")
        r = self.http.post(TOKEN_URL, data={
            "grant_type": "authorization_code",
            "client_id": self.s.etsy_api_key,
            "redirect_uri": self.s.etsy_redirect_uri,
            "code": code,
            "code_verifier": pending["verifier"],
        })
        if r.status_code != 200:
            raise EtsyError(f"Etsy token request failed: {r.status_code} {r.text[:300]}")
        self._save_token(r.json())
        self.pending_path.unlink(missing_ok=True)
        self.url_path.unlink(missing_ok=True)
        log.info("Etsy authorisation completed")

    def _save_token(self, payload: dict) -> None:
        token = {
            "access_token": payload["access_token"],
            "refresh_token": payload["refresh_token"],
            "expires_at": time.time() + int(payload.get("expires_in", 3600)),
        }
        tmp = self.token_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(token), encoding="utf-8")
        tmp.replace(self.token_path)

    def _access_token(self, force_refresh: bool = False) -> str:
        if not self.token_path.exists():
            raise EtsyAuthPending("no Etsy token")
        token = json.loads(self.token_path.read_text(encoding="utf-8"))
        if force_refresh or token["expires_at"] - time.time() < 120:
            r = self.http.post(TOKEN_URL, data={
                "grant_type": "refresh_token",
                "client_id": self.s.etsy_api_key,
                "refresh_token": token["refresh_token"],
            })
            if r.status_code != 200:
                # refresh token expired or revoked: start a fresh authorisation
                self.token_path.unlink(missing_ok=True)
                self.start_authorization()
                raise EtsyAuthPending(f"Etsy token refresh failed ({r.status_code}): new authorisation needed")
            self._save_token(r.json())
            token = json.loads(self.token_path.read_text(encoding="utf-8"))
        return token["access_token"]

    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        url = path if path.startswith("http") else API_BASE + path
        for attempt in range(2):
            headers = {"x-api-key": self._api_key_header,
                       "Authorization": f"Bearer {self._access_token(force_refresh=attempt == 1)}"}
            r = self.http.request(method, url, headers=headers, **kw)
            if r.status_code != 401:
                return r
        return r

    def _ok(self, r: httpx.Response, what: str) -> dict:
        if r.status_code >= 300:
            raise EtsyError(f"{what} failed: {r.status_code} {r.text[:400]}")
        return r.json() if r.content else {}

    # ------------------------------------------------------------------ shop facts
    def _cache(self) -> dict:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _store_cache(self, **values) -> None:
        data = self._cache() | values
        self.cache_path.write_text(json.dumps(data), encoding="utf-8")

    def shop_id(self) -> int:
        cached = self._cache().get("shop_id")
        if cached:
            return int(cached)
        me = self._ok(self._request("GET", "/users/me"), "getMe")
        if not me.get("shop_id"):
            raise EtsyError("this Etsy account has no shop")
        self._store_cache(shop_id=me["shop_id"])
        return int(me["shop_id"])

    def taxonomy_id(self) -> int:
        if self.s.etsy_taxonomy_id:
            return self.s.etsy_taxonomy_id
        cached = self._cache().get("taxonomy_id")
        if cached:
            return int(cached)
        nodes = self._ok(self._request("GET", "/seller-taxonomy/nodes"), "getSellerTaxonomyNodes").get("results", [])
        found = _find_taxonomy(nodes, "digital prints")
        if not found:
            raise EtsyError("taxonomy 'Digital Prints' not found: set ETSY_TAXONOMY_ID")
        self._store_cache(taxonomy_id=found)
        return found

    # ------------------------------------------------------------------ drafts
    def create_draft(self, slug_dir: Path) -> int:
        """Create (or finish) the draft for one Etsy package. Returns the listing id."""
        marker_path = slug_dir / MARKER
        marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.exists() else {}
        if marker.get("status") in {"complete", "skip"}:
            return int(marker.get("listing_id") or 0)
        zips, mockups = package_contents(slug_dir)
        if len(zips) > MAX_FILES:
            raise EtsyError(f"{slug_dir.name}: {len(zips)} zip files, Etsy accepts at most {MAX_FILES}")
        if not mockups:
            raise EtsyError(f"{slug_dir.name}: no mockup images")
        shop = self.shop_id()

        listing_id = marker.get("listing_id")
        if not listing_id:
            text = parse_listing_file(slug_dir / "etsy_listing.txt", self.s.etsy_language)
            listing_id = self._create_listing(shop, text)
            marker = {"listing_id": listing_id, "status": "incomplete", "images": [], "files": []}
            marker_path.write_text(json.dumps(marker, indent=1), encoding="utf-8")

        for rank, img in enumerate(mockups, start=1):
            if img.name in marker["images"]:
                continue
            with img.open("rb") as fh:
                r = self._request("POST", f"/shops/{shop}/listings/{listing_id}/images",
                                  data={"rank": str(rank)}, files={"image": (img.name, fh, "image/jpeg")})
            self._ok(r, f"uploadListingImage {img.name}")
            marker["images"].append(img.name)
            marker_path.write_text(json.dumps(marker, indent=1), encoding="utf-8")

        for rank, z in enumerate(zips, start=1):
            if z.name in marker["files"]:
                continue
            with z.open("rb") as fh:
                r = self._request("POST", f"/shops/{shop}/listings/{listing_id}/files",
                                  data={"name": z.name, "rank": str(rank)},
                                  files={"file": (z.name, fh, "application/zip")})
            self._ok(r, f"uploadListingFile {z.name}")
            marker["files"].append(z.name)
            marker_path.write_text(json.dumps(marker, indent=1), encoding="utf-8")

        other = _OTHER_LANGUAGE.get(self.s.etsy_language.lower())
        if self.s.etsy_translate and other and "translation" not in marker:
            marker["translation"] = self._translate(shop, int(listing_id), slug_dir, other)
            marker_path.write_text(json.dumps(marker, indent=1), encoding="utf-8")

        marker["status"] = "complete"
        marker["completed_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        marker_path.write_text(json.dumps(marker, indent=1), encoding="utf-8")
        log.info("Etsy draft ready", extra={"slug": slug_dir.name, "listing_id": listing_id,
                                            "images": len(mockups), "files": len(zips)})
        return int(listing_id)

    def _translate(self, shop: int, listing_id: int, slug_dir: Path, language: str) -> str:
        """Add the second language as an Etsy translation. A refusal (for example the language is not
        enabled in the shop) is logged and never blocks the draft."""
        try:
            text = parse_listing_file(slug_dir / "etsy_listing.txt", language)
        except (EtsyError, OSError) as exc:
            log.warning("Etsy translation skipped", extra={"slug": slug_dir.name, "reason": str(exc)})
            return "skipped"
        r = self._request("PUT", f"/shops/{shop}/listings/{listing_id}/translations/{language}",
                          data={"title": text.title, "description": text.description, "tags": ",".join(text.tags)})
        if r.status_code >= 300:
            log.warning("Etsy translation refused", extra={"slug": slug_dir.name, "language": language,
                                                           "status": r.status_code, "body": r.text[:300]})
            return "failed"
        return language

    def _create_listing(self, shop: int, text: ListingText) -> int:
        base = {
            "quantity": str(self.s.etsy_quantity),
            "title": text.title,
            "description": text.description,
            "price": f"{self.s.etsy_price:.2f}",
            "who_made": "i_did",
            "is_supply": "false",
            "taxonomy_id": str(self.taxonomy_id()),
            "type": "download",
            "tags": ",".join(text.tags),
        }
        last = ""
        for era in when_made_candidates(self.s.etsy_when_made):
            r = self._request("POST", f"/shops/{shop}/listings", data=base | {"when_made": era})
            if r.status_code < 300:
                return int(r.json()["listing_id"])
            last = f"{r.status_code} {r.text[:400]}"
            if "when_made" not in r.text:
                break
        raise EtsyError(f"createDraftListing failed: {last}")

    def sync(self) -> int:
        """Create drafts for every Etsy package without a finished draft. Returns how many were finished."""
        if not self.ensure_auth():
            return 0
        done = 0
        for slug_dir in sorted(p for p in self.s.etsy_dir.iterdir() if p.is_dir()):
            if not (slug_dir / "etsy_listing.txt").exists():
                continue
            marker = slug_dir / MARKER
            if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("status") in {"complete", "skip"}:
                continue
            try:
                self.create_draft(slug_dir)
                done += 1
            except EtsyAuthPending:
                raise
            except (EtsyError, httpx.HTTPError, OSError) as exc:
                log.error("Etsy draft failed", extra={"slug": slug_dir.name, "error": str(exc)})
        return done


def _find_taxonomy(nodes: list[dict], name: str) -> int | None:
    for n in nodes:
        if str(n.get("name", "")).strip().lower() == name:
            return int(n["id"])
        found = _find_taxonomy(n.get("children") or [], name)
        if found:
            return found
    return None
