"""Microstock upload over FTPS (ftplib.FTP_TLS) or SFTP (paramiko), with DRY_RUN and size verification."""

from __future__ import annotations

import ftplib
import logging
import posixpath
import socket
import ssl
import time
from pathlib import Path
from typing import Any

from .config import UploadTarget

log = logging.getLogger(__name__)


class UploadError(Exception):
    pass


class _ReusedSessionFTPTLS(ftplib.FTP_TLS):
    """FTP_TLS that reuses the control-channel TLS session on the data channel.

    Many FTPS servers (vsftpd/pure-ftpd with `require_ssl_reuse`) reject data connections
    otherwise, which plain ftplib does not do.
    """

    def ntransfercmd(self, cmd: str, rest: Any = None) -> tuple[socket.socket, int | None]:
        conn, size = ftplib.FTP.ntransfercmd(self, cmd, rest)
        if self._prot_p:  # type: ignore[attr-defined]
            conn = self.context.wrap_socket(
                conn, server_hostname=self.host, session=self.sock.session  # type: ignore[union-attr]
            )
        return conn, size


def _upload_ftps(target: UploadTarget, local: Path, remote_name: str, timeout: int) -> None:
    ctx = ssl.create_default_context()
    ftp = _ReusedSessionFTPTLS(context=ctx, timeout=timeout)
    try:
        ftp.connect(target.host, target.port)
        ftp.login(target.username, target.password)
        ftp.prot_p()
        ftp.set_pasv(True)
        if target.remote_dir and target.remote_dir != "/":
            ftp.cwd(target.remote_dir)
        with local.open("rb") as fh:
            ftp.storbinary(f"STOR {remote_name}", fh, blocksize=256 * 1024)
        try:
            ftp.voidcmd("TYPE I")
            remote_size = ftp.size(remote_name)
        except ftplib.error_perm:
            remote_size = None  # SIZE not supported by every server
        if remote_size is not None and remote_size != local.stat().st_size:
            raise UploadError(f"size mismatch after FTPS upload: remote={remote_size} local={local.stat().st_size}")
    finally:
        try:
            ftp.quit()
        except (ftplib.Error, OSError):
            ftp.close()


def _upload_sftp(target: UploadTarget, local: Path, remote_name: str, timeout: int, known_hosts: Path) -> None:
    import paramiko

    client = paramiko.SSHClient()
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    if known_hosts.exists():
        client.load_host_keys(str(known_hosts))
    # Trust-on-first-use: the key is pinned in known_hosts on first connection, later changes are rejected.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            target.host, port=target.port, username=target.username, password=target.password,
            timeout=timeout, banner_timeout=timeout, auth_timeout=timeout,
            look_for_keys=False, allow_agent=False,
        )
        client.save_host_keys(str(known_hosts))
        sftp = client.open_sftp()
        try:
            remote_path = posixpath.join(target.remote_dir or "/", remote_name) if target.remote_dir not in ("", "/") else remote_name
            attrs = sftp.put(str(local), remote_path, confirm=True)
            if attrs.st_size is not None and attrs.st_size != local.stat().st_size:
                raise UploadError(f"size mismatch after SFTP upload: remote={attrs.st_size}")
        finally:
            sftp.close()
    except paramiko.BadHostKeyException as exc:
        raise UploadError(f"SFTP host key changed for {target.host}! Refusing to upload: {exc}") from exc
    finally:
        client.close()


def upload_file(
    target: UploadTarget,
    local: Path,
    remote_name: str,
    *,
    dry_run: bool,
    max_retries: int = 3,
    timeout: int = 120,
    known_hosts: Path,
) -> str:
    """Upload one file; returns 'dry_run' or 'uploaded'. Raises UploadError after retries."""
    if not local.is_file():
        raise UploadError(f"local file missing: {local}")
    size = local.stat().st_size
    if dry_run:
        log.info(
            "[DRY-RUN] upload simulated",
            extra={"platform": target.name, "protocol": target.protocol, "host": target.host,
                   "remote_dir": target.remote_dir, "file": remote_name, "bytes": size, "user": target.username},
        )
        return "dry_run"
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.monotonic()
            if target.protocol == "ftps":
                _upload_ftps(target, local, remote_name, timeout)
            elif target.protocol == "sftp":
                _upload_sftp(target, local, remote_name, timeout, known_hosts)
            else:
                raise UploadError(f"unknown protocol {target.protocol}")
            log.info(
                "upload completed",
                extra={"platform": target.name, "file": remote_name, "bytes": size, "seconds": round(time.monotonic() - t0, 1)},
            )
            return "uploaded"
        except ftplib.error_perm as exc:  # auth/permission errors are not transient
            raise UploadError(f"{target.name}: permission error: {exc}") from exc
        except UploadError as exc:
            if "host key changed" in str(exc):
                raise
            last = exc
        except Exception as exc:
            last = exc
        delay = min(300, 10 * 2 ** (attempt - 1))
        log.warning("upload failed", extra={"platform": target.name, "attempt": attempt, "retry_in_s": delay, "error": str(last)})
        if attempt < max_retries:
            time.sleep(delay)
    raise UploadError(f"{target.name}: upload failed after {max_retries} attempts: {last}")
