"""Optional Telegram notifications (never raise: a notification must not break the pipeline)."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, timeout: float = 15.0) -> None:
        self.enabled = bool(token and chat_id)
        self._url = f"https://api.telegram.org/bot{token}/sendMessage" if self.enabled else ""
        self._chat_id = chat_id
        self._timeout = timeout

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            resp = httpx.post(
                self._url,
                json={"chat_id": self._chat_id, "text": text[:4000], "disable_web_page_preview": True},
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                log.warning("telegram notification rejected", extra={"status": resp.status_code, "body": resp.text[:200]})
                return False
            return True
        except httpx.HTTPError as exc:
            log.warning("telegram notification failed", extra={"error": str(exc)})
            return False
