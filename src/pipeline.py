"""Orchestrates one photo end-to-end: load -> AI -> upscale -> prints -> zips -> mockups -> microstock."""

from __future__ import annotations

import gc
import logging
import re
import shutil
import traceback
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .ai_analyzer import AIAnalysis, AIQuotaExhaustedError, GeminiAnalyzer
from .config import Settings
from .database import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    UPLOAD_DRY_RUN,
    UPLOAD_FAILED,
    UPLOAD_UPLOADED,
    Database,
    Job,
)
from .ftp_uploader import UploadError, upload_file
from .image_loader import ImageLoadError, LoadedImage, load_image, make_preview_jpeg
from .metadata_writer import export_microstock_jpeg, write_metadata
from .mockup_generator import generate_mockups
from .notifier import TelegramNotifier
from .packaging import PackagingError, TooManyPartsError, package_files
from .print_generator import FORMATS, POSTER_FORMAT_KEYS, PrintOutputs, generate_prints
from .smart_crop import find_focus
from .upscaler import upscale_if_needed

log = logging.getLogger(__name__)

PERMANENT_ERRORS = (ImageLoadError, PackagingError)


def _close_previews(prints: PrintOutputs) -> None:
    for im in (prints.fullbleed_preview, prints.poster_preview, *prints.crop_previews.values()):
        im.close()


def slugify(text: str, max_len: int = 60) -> str:
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", norm).strip("-").lower()
    if len(slug) > max_len:
        slug = slug[:max_len].rsplit("-", 1)[0]
    return slug or "photo"


def make_slug(ai: AIAnalysis, context: str, sha256: str) -> str:
    seed = " ".join(p for p in (ai.poster_title, ai.location_place, context) if p)
    words = slugify(seed).split("-")
    unique: list[str] = []
    for w in words:
        if w not in unique:
            unique.append(w)
    return f"{'-'.join(unique[:6]) or 'photo'}-{sha256[:8]}"


@dataclass
class JobResult:
    slug: str
    etsy_dir: Path
    zips: list[Path]
    mockups: list[Path]
    microstock_file: Path
    commercial_safe: bool
    upscale_method: str
    upload_states: dict[str, str] = field(default_factory=dict)


# ----------------------------------------------------------------------------- texts
def _has_posters(slug_zips: list[str]) -> bool:
    return any(z.startswith("prints_gallery_poster") for z in slug_zips)


def _what_you_get_en(slug_zips: list[str], orientation: str, dpi: int) -> str:
    posters = _has_posters(slug_zips)
    extra = f" + {len(POSTER_FORMAT_KEYS)} Gallery Poster editions" if posters else ""
    lines = [f"• {len(FORMATS)} full-bleed high-resolution JPEG files{extra}, {dpi} DPI, sRGB"]
    for f in FORMATS:
        w, h = f.pixels(orientation)
        lines.append(f"  - {f.ratio_label} ({w}x{h} px) → prints {', '.join(f.sizes_in)}{'' if f.key == 'iso_A_series' else ' in'} / {', '.join(f.sizes_cm)} cm")
    if posters:
        lines.append("• Gallery Poster edition (warm white paper margins + typography): 2:3 and ISO A")
    lines.append(f"• Delivered as ZIP: {', '.join(slug_zips)}")
    lines.append("• INSTANT DIGITAL DOWNLOAD — no physical item will be shipped. Frame not included.")
    lines.append("• Colours may vary slightly depending on monitor and printer. For personal use only.")
    return "\n".join(lines)


def _what_you_get_it(slug_zips: list[str], orientation: str, dpi: int) -> str:
    posters = _has_posters(slug_zips)
    extra = f" + {len(POSTER_FORMAT_KEYS)} versioni Gallery Poster" if posters else ""
    lines = [f"• {len(FORMATS)} file JPEG ad alta risoluzione a stampa piena{extra}, {dpi} DPI, sRGB"]
    for f in FORMATS:
        w, h = f.pixels(orientation)
        lines.append(f"  - {f.ratio_label} ({w}x{h} px) → stampe {', '.join(f.sizes_cm)} cm / {', '.join(f.sizes_in)}{'' if f.key == 'iso_A_series' else ' in'}")
    if posters:
        lines.append("• Versione Gallery Poster (margini carta bianco caldo + tipografia): 2:3 e ISO A")
    lines.append(f"• Consegna in ZIP: {', '.join(slug_zips)}")
    lines.append("• DOWNLOAD DIGITALE IMMEDIATO — nessun prodotto fisico verrà spedito. Cornice non inclusa.")
    lines.append("• I colori possono variare leggermente in base a monitor e stampante. Solo per uso personale.")
    return "\n".join(lines)


def build_etsy_listing(ai: AIAnalysis, slug: str, zips: list[Path], orientation: str, dpi: int) -> str:
    zip_names = [z.name for z in zips]
    sep = "=" * 72

    def section(lang: str, header: str, extra: str) -> str:
        t = getattr(ai, f"etsy_{lang}")
        return (
            f"{sep}\n{header}\n{sep}\n\n"
            f"TITLE / TITOLO ({len(t.title)}/140):\n{t.title}\n\n"
            f"DESCRIPTION / DESCRIZIONE:\n{t.description}\n\n{extra}\n\n"
            f"TAGS (13):\n{', '.join(t.tags)}\n"
        )

    return (
        f"ETSY LISTING — {slug}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Gallery Poster: {ai.poster_title} — {ai.poster_subtitle}\n\n"
        + section("en", "ENGLISH", "— WHAT YOU GET —\n" + _what_you_get_en(zip_names, orientation, dpi))
        + "\n"
        + section("it", "ITALIANO", "— COSA RICEVI —\n" + _what_you_get_it(zip_names, orientation, dpi))
    )


# ----------------------------------------------------------------------------- pipeline
class Pipeline:
    def __init__(self, settings: Settings, db: Database, notifier: TelegramNotifier | None = None) -> None:
        self.s = settings
        self.db = db
        self.notifier = notifier or TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
        self._analyzer: GeminiAnalyzer | None = None
        self._etsy = None

    @property
    def analyzer(self) -> GeminiAnalyzer:
        if self._analyzer is None:
            s = self.s
            self._analyzer = GeminiAnalyzer(
                api_key=s.gemini_api_key, model=s.gemini_model, min_interval=s.gemini_min_interval_seconds,
                max_retries=s.gemini_max_retries, backoff_base=s.gemini_backoff_base_seconds,
                backoff_max=s.gemini_backoff_max_seconds, timeout_seconds=s.gemini_timeout_seconds,
                thinking_budget=s.gemini_thinking_budget,
            )
        return self._analyzer

    # ------------------------------------------------------------------ public
    def run_job(self, job: Job) -> None:
        src = Path(job.src_path)
        ctx = {"job_id": job.id, "file": src.name, "attempt": job.attempts}
        log.info("job started", extra=ctx)
        if not src.exists():
            self.db.update_job(job.id, status=STATUS_FAILED, error="source file no longer exists")
            log.error("source missing, job failed", extra=ctx)
            return
        try:
            result = self._process(job, src)
        except AIQuotaExhaustedError as exc:
            delay = self.s.gemini_quota_cooldown_minutes * 60
            self.db.defer_job(job.id, delay, f"quota: {exc}", refund_attempt=True)
            log.warning("Gemini quota exhausted, job deferred", extra={**ctx, "retry_in_min": self.s.gemini_quota_cooldown_minutes})
            return
        except Exception as exc:
            self._handle_failure(job, src, exc)
            return
        finally:
            self._cleanup_work(job)
            gc.collect()

        processed = self._move_original(src, self.s.processed_dir, job.sha256)
        self.db.update_job(job.id, status=STATUS_COMPLETED, error=None, src_path=str(processed))
        log.info("job completed", extra={**ctx, "slug": result.slug, "upscale": result.upscale_method})
        self._notify_done(result)

    def sync_etsy_drafts(self) -> int:
        """Create Etsy drafts for finished packages (never fails the worker loop)."""
        try:
            if self._etsy is None:
                from .etsy_publisher import EtsyClient

                self._etsy = EtsyClient(self.s)
            return self._etsy.sync()
        except Exception as exc:  # Etsy problems must never stop the pipeline
            log.warning("Etsy sync skipped", extra={"reason": str(exc)})
            return 0

    def retry_pending_uploads(self) -> int:
        """Upload files whose transfer previously failed or ran in DRY_RUN (no-op while DRY_RUN)."""
        if self.s.dry_run:
            log.info("DRY_RUN active: pending uploads left untouched")
            return 0
        targets = {t.name: t for t in self.s.upload_targets()}
        for name in targets:  # photos processed before this platform was enabled
            for job_id, micro in self.db.jobs_missing_upload(name):
                if Path(micro).exists():
                    self.db.ensure_upload(job_id, name, Path(micro), Path(micro).name)
                    log.info("upload queued for earlier photo", extra={"platform": name, "file": Path(micro).name})
        done = 0
        for up in self.db.uploads_needing_transfer():
            target = targets.get(up.platform)
            if target is None:
                continue
            local = Path(up.local_path)
            try:
                state = upload_file(
                    target, local, up.remote_name, dry_run=False,
                    max_retries=self.s.upload_max_retries, timeout=self.s.upload_timeout_seconds,
                    known_hosts=self.s.known_hosts_path,
                )
                self.db.set_upload_status(up.id, state)
                done += 1
            except UploadError as exc:
                self.db.set_upload_status(up.id, UPLOAD_FAILED, str(exc))
                continue
            self._finalise_microstock_location(up.job_id)
        return done

    # ------------------------------------------------------------------ steps
    def _prints_and_zips(
        self, loaded: LoadedImage, focus, slug: str, ai: AIAnalysis, work: Path, stage: Path
    ) -> tuple[PrintOutputs, list[Path]]:
        """Render the prints and zip them into at most `etsy_max_files` archives (Etsy's per-listing limit).

        Very detailed photos (foliage, water, gravel) produce heavier JPEGs. When the zips would exceed the
        limit, the prints are rendered again at a slightly lower JPEG quality, never below `jpeg_quality_min`.
        If even that does not fit, the Gallery Poster set is left out: the full-bleed set always fits,
        because every print is smaller than one zip.
        """
        s = self.s
        quality = s.jpeg_quality_prints
        while True:
            prints = generate_prints(
                loaded.image, orientation=loaded.orientation, focus=focus, slug=slug, out_dir=work,
                title=ai.poster_title, subtitle=ai.poster_subtitle, fonts_dir=str(s.fonts_dir),
                dpi=s.print_dpi, quality=quality, min_quality=s.jpeg_quality_min,
                max_file_bytes=s.max_print_file_bytes,
            )
            full: list[Path] = []
            try:
                full = package_files(
                    prints.fullbleed_files, stage, "prints_fullbleed", s.etsy_zip_max_bytes, s.etsy_max_files
                )
                slots_left = s.etsy_max_files - len(full)
                if slots_left < 1:
                    raise TooManyPartsError("no archive slot left for the gallery posters")
                posters = package_files(
                    prints.poster_files, stage, "prints_gallery_poster", s.etsy_zip_max_bytes, slots_left
                )
                if quality != s.jpeg_quality_prints:
                    log.info("prints re-encoded to respect the Etsy file limit", extra={"quality": quality})
                return prints, full + posters
            except TooManyPartsError as exc:
                if quality > s.jpeg_quality_min:
                    _close_previews(prints)
                    gc.collect()
                    quality = max(s.jpeg_quality_min, quality - s.jpeg_quality_step)
                    log.info("too many zips for Etsy, retrying at lower JPEG quality",
                             extra={"reason": str(exc), "quality": quality})
                    continue
                if not full:
                    _close_previews(prints)
                    raise
                for stale in stage.glob("prints_gallery_poster*.zip"):
                    stale.unlink()
                log.warning("gallery posters left out to respect the Etsy file limit",
                            extra={"zips": len(full), "limit": s.etsy_max_files})
                return prints, full

    def _process(self, job: Job, src: Path) -> JobResult:
        s = self.s
        loaded: LoadedImage = load_image(src, s.input_dir, s.analog_keyword_list)
        try:
            self.db.update_job(job.id, is_analog=int(loaded.is_analog_film))
            ai = self._analyse(job, loaded)
            slug = job.slug or make_slug(ai, loaded.context_text, job.sha256)
            self.db.update_job(
                job.id, slug=slug, commercial_safe=int(ai.microstock_commercial_safe), safety_notes=ai.safety_notes
            )

            # Microstock export uses the developed original (agencies reject upscaled images).
            micro_path = self._export_microstock(job, loaded, ai, slug, src)

            upscaled, method = upscale_if_needed(
                loaded.image, is_analog=loaded.is_analog_film, threshold=s.upscale_threshold_px,
                target_long_side=s.upscale_target_long_side, engine=s.upscale_engine,
                model_path=s.upscale_model_path, tile_size=s.upscale_tile_size,
                overlap=s.upscale_tile_overlap, threads=s.onnx_threads,
            )
            if upscaled is not loaded.image:
                loaded.image.close()
                loaded.image = upscaled
                gc.collect()

            focus = find_focus(loaded.image)
            log.info("focus computed", extra={"cx": round(focus.cx, 3), "cy": round(focus.cy, 3), "method": focus.method})

            work = s.work_dir / job.sha256
            if work.exists():
                shutil.rmtree(work)
            stage = work / slug
            prints, zips = self._prints_and_zips(loaded, focus, slug, ai, work, stage)
            loaded.close()
            gc.collect()

            try:
                mockups = generate_mockups(
                    prints.fullbleed_preview, prints.poster_preview, prints.crop_previews,
                    loaded.orientation, stage / "mockups", str(s.fonts_dir), s.print_dpi,
                )
            finally:
                _close_previews(prints)
            (stage / "etsy_listing.txt").write_text(
                build_etsy_listing(ai, slug, zips, loaded.orientation, s.print_dpi), encoding="utf-8"
            )

            final_dir = s.etsy_dir / slug
            if final_dir.exists():
                shutil.rmtree(final_dir)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stage), final_dir)
            zips = [final_dir / z.name for z in zips]
            mockups = [final_dir / "mockups" / m.name for m in mockups]
            log.info("etsy package ready", extra={"dir": str(final_dir), "zips": len(zips)})

            upload_states = self._upload_microstock(job, micro_path, ai)
            return JobResult(
                slug=slug, etsy_dir=final_dir, zips=zips, mockups=mockups,
                microstock_file=Path(self.db.get_microstock_path(job.id) or micro_path),
                commercial_safe=ai.microstock_commercial_safe, upscale_method=method,
                upload_states=upload_states,
            )
        finally:
            loaded.close()

    def _analyse(self, job: Job, loaded: LoadedImage) -> AIAnalysis:
        cached = job.ai_data
        if cached:
            try:
                log.info("using cached AI analysis", extra={"job_id": job.id})
                return AIAnalysis.model_validate(cached)
            except ValueError:
                log.warning("cached AI analysis invalid, calling Gemini again", extra={"job_id": job.id})
        preview = make_preview_jpeg(loaded.image, self.s.gemini_preview_long_side, 80)
        ai = self.analyzer.analyse(
            preview, loaded.context_text, loaded.exif, loaded.is_analog_film, loaded.orientation
        )
        self.db.update_job(job.id, ai_json=ai.model_dump_json())
        return ai

    def _export_microstock(self, job: Job, loaded: LoadedImage, ai: AIAnalysis, slug: str, src: Path) -> Path:
        s = self.s
        if ai.microstock_commercial_safe:
            path = s.microstock_pending_dir / f"{slug}.jpg"
        else:
            reason = "-".join(ai.safety_flags) or "flagged"
            path = s.microstock_review_dir / f"{slug}__EDITORIAL_REVIEW_{reason}.jpg"
            log.warning("not commercially safe: sent to editorial review", extra={"file": path.name, "notes": ai.safety_notes})
        export_microstock_jpeg(loaded.image, path, s.microstock_jpeg_quality, s.microstock_min_megapixels)
        write_metadata(
            path, ai.microstock_title, ai.microstock_description, ai.microstock_keywords,
            original=src, keep_gps=s.microstock_keep_gps, author=s.author_name, copyright_notice=s.copyright_notice,
        )
        self.db.update_job(job.id, microstock_path=str(path))
        return path

    def _upload_microstock(self, job: Job, micro_path: Path, ai: AIAnalysis) -> dict[str, str]:
        if not ai.microstock_commercial_safe:
            return {}
        targets = self.s.upload_targets()
        if not targets:
            log.info("no upload target enabled: microstock file left in pending", extra={"file": micro_path.name})
            return {}
        states: dict[str, str] = {}
        for target in targets:
            up = self.db.ensure_upload(job.id, target.name, micro_path, micro_path.name)
            if up.status == UPLOAD_UPLOADED:
                states[target.name] = UPLOAD_UPLOADED
                continue
            try:
                state = upload_file(
                    target, micro_path, micro_path.name, dry_run=self.s.dry_run,
                    max_retries=self.s.upload_max_retries, timeout=self.s.upload_timeout_seconds,
                    known_hosts=self.s.known_hosts_path,
                )
                self.db.set_upload_status(up.id, state)
                states[target.name] = state
            except UploadError as exc:
                self.db.set_upload_status(up.id, UPLOAD_FAILED, str(exc))
                states[target.name] = UPLOAD_FAILED
                log.error("microstock upload failed (will be retried)", extra={"platform": target.name, "error": str(exc)})
        self._finalise_microstock_location(job.id)
        return states

    def _finalise_microstock_location(self, job_id: int) -> None:
        """Move the microstock JPEG to microstock_uploaded/ once every enabled target has it."""
        uploads = self.db.uploads_for_job(job_id)
        enabled = {t.name for t in self.s.upload_targets()}
        relevant = [u for u in uploads if u.platform in enabled]
        if not relevant or any(u.status != UPLOAD_UPLOADED for u in relevant):
            return
        current = Path(relevant[0].local_path)
        if not current.exists() or current.parent == self.s.microstock_uploaded_dir:
            return
        self.s.microstock_uploaded_dir.mkdir(parents=True, exist_ok=True)
        target = self.s.microstock_uploaded_dir / current.name
        shutil.move(str(current), target)
        for u in uploads:
            self.db.update_upload_path(u.id, target)
        self.db.update_job(job_id, microstock_path=str(target))
        log.info("microstock file moved to uploaded", extra={"file": target.name})

    # ------------------------------------------------------------------ housekeeping
    def _move_original(self, src: Path, dest_root: Path, sha256: str) -> Path:
        try:
            rel = src.relative_to(self.s.input_dir)
        except ValueError:
            rel = Path(src.name)
        target = dest_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(f"{target.stem}_{sha256[:8]}{target.suffix}")
        shutil.move(str(src), target)
        return target

    def _cleanup_work(self, job: Job) -> None:
        work = self.s.work_dir / job.sha256
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)

    def _handle_failure(self, job: Job, src: Path, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        permanent = isinstance(exc, PERMANENT_ERRORS) or (isinstance(exc, MemoryError) and job.attempts >= 2)
        if not permanent and job.attempts < self.s.max_job_attempts:
            delay = self.s.retry_delay_minutes * 60 * job.attempts
            self.db.defer_job(job.id, delay, f"{type(exc).__name__}: {exc}")
            log.warning(
                "job failed, will retry",
                extra={"job_id": job.id, "attempt": job.attempts, "retry_in_min": delay // 60, "error": str(exc)[:300]},
            )
            return
        failed_path = src
        if src.exists():
            failed_path = self._move_original(src, self.s.failed_dir, job.sha256)
        log_path = failed_path.with_name(failed_path.name + ".error.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"time: {datetime.now().isoformat()}\njob_id: {job.id}\nsha256: {job.sha256}\n"
            f"original: {job.src_path}\nattempts: {job.attempts}\n"
            f"category: {'permanent' if permanent else 'max attempts reached'}\n\n{tb}",
            encoding="utf-8",
        )
        self.db.update_job(job.id, status=STATUS_FAILED, error=f"{type(exc).__name__}: {exc}"[:2000], src_path=str(failed_path))
        log.error("job failed permanently", extra={"job_id": job.id, "error": str(exc)[:300], "log": str(log_path)})
        self.notifier.send(f"❌ Pipeline FAILED: {src.name}\n{type(exc).__name__}: {str(exc)[:300]}")

    def _notify_done(self, r: JobResult) -> None:
        if r.commercial_safe:
            ups = ", ".join(f"{k}: {v}" for k, v in r.upload_states.items()) or "no target enabled"
            micro = f"Microstock: {ups}"
        else:
            micro = "Microstock: ⚠️ editorial review (not commercially safe)"
        dry = " [DRY_RUN]" if self.s.dry_run and any(v == UPLOAD_DRY_RUN for v in r.upload_states.values()) else ""
        self.notifier.send(
            f"✅ Photo ready{dry}: {r.slug}\nEtsy: {len(r.zips)} zip, {len(r.mockups)} mockups\n{micro}\nUpscale: {r.upscale_method}"
        )

