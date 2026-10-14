"""Fetch the content behind links found in PDFs.

Handles four kinds of link:
    * web pages      -> requests + trafilatura, Playwright fallback for JS-heavy pages
    * linked PDFs    -> downloaded and handed back to the pipeline for full extraction
    * audio / video  -> downloaded and transcribed with Whisper
    * YouTube        -> audio downloaded with yt-dlp, transcribed, plus title/description
"""
import hashlib
import logging
import ssl
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

from . import config, media
from .models import Document

log = logging.getLogger(__name__)

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"}
SKIP_HOSTS = {"facebook.com", "www.facebook.com", "twitter.com", "x.com", "instagram.com",
              "www.instagram.com", "wa.me", "api.whatsapp.com"}


@dataclass
class ScrapeResult:
    url: str
    kind: str                                   # html | pdf | audio | video | image | youtube
    docs: list = field(default_factory=list)    # list[Document]
    pdf_path: Path | None = None                # set when the link is a PDF
    links: list = field(default_factory=list)   # same-site links found on an HTML page


class _LenientTLSAdapter(HTTPAdapter):
    """For SSL_FALLBACK only: accepts expired / mismatched certificates and old servers that
    need legacy TLS renegotiation. Used only after a normal, verified request has failed."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0x4)
        try:
            ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
        except ssl.SSLError:
            pass
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def _use_os_certificates():
    """Verify certificates with the operating system's store (Windows/macOS). Fixes
    'unable to get local issuer certificate' on many .gov.in sites that don't send their
    intermediate certificate - Windows fetches it automatically, Python's own store can't."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass


def _reason(exc: Exception) -> str:
    """Short, plain-English reason for a failed fetch."""
    if isinstance(exc, requests.exceptions.SSLError):
        return "SSL certificate problem on the website"
    if isinstance(exc, requests.exceptions.Timeout):
        return "website timed out"
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        code = exc.response.status_code
        if code in (401, 403):
            return f"website refused access ({code})"
        if code == 404:
            return "page not found (404)"
        if code >= 500:
            return f"website error ({code})"
        return f"HTTP {code}"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "could not connect to website"
    return str(exc)[:120]


class Scraper:
    def __init__(self):
        _use_os_certificates()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT,
                                     "Accept-Language": "en-IN,en;q=0.9,ta;q=0.8"})
        self.download_dir = config.CACHE_DIR / "downloads"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._robots: dict = {}
        self._last_hit: dict = {}
        self._seen: set = set()
        self._pw = None
        self._browser = None
        self._pw_failed = False
        self._lenient = None                 # session used only for SSL_FALLBACK
        self.failures: list = []             # [(url, reason)] for the ingestion report

    # ------------------------------------------------------------------ public
    def fetch(self, url: str) -> ScrapeResult | None:
        """Fetch one URL. Returns None if skipped, already seen, blocked or failed."""
        if url in self._seen:
            return None
        self._seen.add(url)
        host = urlparse(url).netloc.lower()
        if host in SKIP_HOSTS:
            return None
        if "." not in host.split(":")[0]:
            # e.g. "https://www" - a link that was broken across two lines in the PDF
            self.failures.append((url, "incomplete link in the PDF"))
            return None
        try:
            if host in YOUTUBE_HOSTS:
                return self._youtube(url)
            if not self._allowed(url):
                log.info("robots.txt disallows %s - skipped", url)
                self.failures.append((url, "site's robots.txt disallows scraping"))
                return None
            return self._http(url)
        except Exception as exc:
            reason = _reason(exc)
            log.warning("Failed to fetch %s: %s", url, reason)
            self.failures.append((url, reason))
            return None

    def close(self):
        try:
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
        self._browser = self._pw = None
        self.session.close()

    # ----------------------------------------------------------------- helpers
    def _allowed(self, url: str) -> bool:
        if not config.RESPECT_ROBOTS:
            return True
        parts = urlparse(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                r = self._get(base + "/robots.txt", timeout=10)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception:
                rp.parse([])  # unreachable robots.txt -> allow
            self._robots[base] = rp
        return self._robots[base].can_fetch(config.USER_AGENT, url)

    def _throttle(self, url: str):
        host = urlparse(url).netloc
        wait = config.REQUEST_DELAY - (time.time() - self._last_hit.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        self._last_hit[host] = time.time()

    def _target_path(self, url: str, suffix: str) -> Path:
        digest = hashlib.sha1(url.encode()).hexdigest()[:16]
        return self.download_dir / f"{digest}{suffix}"

    def _download(self, resp: requests.Response, dest: Path) -> Path:
        limit = config.MAX_DOWNLOAD_MB * 1024 * 1024
        size = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                size += len(chunk)
                if size > limit:
                    fh.close()
                    dest.unlink(missing_ok=True)
                    raise ValueError(f"download larger than {config.MAX_DOWNLOAD_MB} MB")
                fh.write(chunk)
        return dest

    # ------------------------------------------------------------------- HTTP
    def _get(self, url: str, **kwargs) -> requests.Response:
        """GET with normal certificate checks; if the site's certificate is broken and
        SSL_FALLBACK is on, retry once without verification (public pages only - nothing is
        sent to the site except the request)."""
        try:
            return self.session.get(url, **kwargs)
        except requests.exceptions.SSLError:
            if not config.SSL_FALLBACK:
                raise
            if self._lenient is None:
                self._lenient = requests.Session()
                self._lenient.headers.update(self.session.headers)
                self._lenient.mount("https://", _LenientTLSAdapter())
            log.warning("Certificate problem on %s - retrying without verification (SSL_FALLBACK)",
                        urlparse(url).netloc)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self._lenient.get(url, verify=False, **kwargs)

    def _http(self, url: str) -> ScrapeResult | None:
        self._throttle(url)
        resp = self._get(url, timeout=config.REQUEST_TIMEOUT, stream=True, allow_redirects=True)
        if resp.status_code in (401, 403) and config.ENABLE_PLAYWRIGHT:
            # Many government sites refuse non-browser clients; a real browser often gets through.
            rendered = self._render(url)
            if rendered:
                title, text = self._extract_html(rendered, url)
                if text:
                    meta = {"source": title or url, "url": url, "source_type": "web",
                            "content_type": "web_page"}
                    return ScrapeResult(url, "html", [Document(text, meta)],
                                        links=self._same_site_links(rendered, url))
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        ext = Path(urlparse(resp.url).path).suffix.lower()
        kind = media.media_kind(ext)

        if "application/pdf" in ctype or kind == "pdf":
            dest = self._download(resp, self._target_path(url, ".pdf"))
            return ScrapeResult(url, "pdf", pdf_path=dest)

        if ctype.startswith(("audio/", "video/")) or kind in ("audio", "video"):
            kind = "audio" if (ctype.startswith("audio/") or kind == "audio") else "video"
            dest = self._download(resp, self._target_path(url, ext or (".mp3" if kind == "audio" else ".mp4")))
            text = media.transcribe(dest)
            docs = [Document(text, {"source": url, "url": url, "source_type": "web",
                                    "content_type": f"{kind}_transcript"})] if text else []
            return ScrapeResult(url, kind, docs)

        if ctype.startswith("image/") or kind == "image":
            dest = self._download(resp, self._target_path(url, ext or ".png"))
            text = media.ocr_image(dest)
            docs = [Document(text, {"source": url, "url": url, "source_type": "web",
                                    "content_type": "image_ocr"})] if len(text) >= config.MIN_OCR_TEXT else []
            return ScrapeResult(url, "image", docs)

        if ctype and not any(t in ctype for t in ("html", "text", "xml")):
            log.info("Unsupported content type %s at %s - skipped", ctype, url)
            return None

        resp.encoding = resp.encoding or resp.apparent_encoding
        html = resp.text
        title, text = self._extract_html(html, url)
        if len(text) < config.MIN_HTML_TEXT and config.ENABLE_PLAYWRIGHT:
            rendered = self._render(url)
            if rendered:
                title2, text2 = self._extract_html(rendered, url)
                if len(text2) > len(text):
                    title, text, html = title2 or title, text2, rendered
        links = self._same_site_links(html, resp.url)
        docs = []
        if text:
            meta = {"source": title or url, "url": url, "source_type": "web", "content_type": "web_page"}
            docs.append(Document(text, meta))
        if not docs and not links:
            return None
        return ScrapeResult(url, "html", docs, links=links)

    @staticmethod
    def _same_site_links(html: str, base_url: str) -> list:
        """Links on the page that stay on the same website (menus, scheme pages, PDFs)."""
        try:
            import lxml.html

            tree = lxml.html.fromstring(html)
            tree.make_links_absolute(base_url, resolve_base_href=True)
        except Exception:
            return []
        host = urlparse(base_url).netloc.lower().removeprefix("www.")
        skip_ext = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".zip", ".rar",
                    ".xls", ".xlsx", ".doc", ".docx", ".ppt", ".pptx", ".ico")
        out, seen = [], set()
        for el, attr, link, _ in tree.iterlinks():
            if el.tag != "a" or attr != "href":
                continue
            link = link.split("#", 1)[0].strip()
            parts = urlparse(link)
            if parts.scheme not in ("http", "https"):
                continue
            if parts.netloc.lower().removeprefix("www.") != host:
                continue
            if parts.path.lower().endswith(skip_ext) or link.rstrip("/") == base_url.rstrip("/"):
                continue
            if link not in seen:
                seen.add(link)
                out.append(link)
        return out

    @staticmethod
    def _extract_html(html: str, url: str) -> tuple[str, str]:
        import trafilatura

        text = trafilatura.extract(html, url=url, include_tables=True, include_links=False,
                                   favor_recall=True) or ""
        if not text:
            try:
                text = trafilatura.html2txt(html) or ""
            except Exception:
                text = ""
        title = ""
        try:
            meta = trafilatura.extract_metadata(html, default_url=url)
            title = (meta.title if meta else "") or ""
        except Exception:
            pass
        return title.strip(), media.clean_text(text)

    def _start_browser(self) -> bool:
        """Start headless Chromium once. If it can't start, don't try again this run."""
        if self._browser is not None:
            return True
        if self._pw_failed:
            return False
        try:
            from playwright.sync_api import sync_playwright

            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=True, args=["--ignore-certificate-errors"]
                                                     if config.SSL_FALLBACK else [])
            return True
        except Exception as exc:
            self._pw_failed = True
            if self._pw is not None:          # stop it, or every later call fails with
                try:                          # "Sync API inside the asyncio loop"
                    self._pw.stop()
                except Exception:
                    pass
                self._pw = None
            hint = (" Run:  playwright install chromium" if "Executable doesn't exist" in str(exc)
                    or "playwright install" in str(exc) else "")
            log.warning("Headless browser unavailable - JavaScript pages will be skipped.%s (%s)",
                        hint, str(exc).splitlines()[0][:150])
            return False

    def _render(self, url: str) -> str:
        """Render a JavaScript-heavy page with headless Chromium (real browser user agent)."""
        if not self._start_browser():
            return ""
        try:
            context = self._browser.new_context(locale="en-IN",
                                                ignore_https_errors=config.SSL_FALLBACK)
            page = context.new_page()
            page.route("**/*", lambda route: route.abort()
                       if route.request.resource_type in ("image", "font", "media")
                       else route.continue_())
            page.goto(url, wait_until="domcontentloaded", timeout=config.REQUEST_TIMEOUT * 1000)
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            html = page.content()
            context.close()
            return html
        except Exception as exc:
            log.warning("Playwright could not render %s: %s", url, exc)
            return ""

    # ----------------------------------------------------------------- YouTube
    def _youtube(self, url: str) -> ScrapeResult | None:
        try:
            import yt_dlp
        except ImportError:
            log.warning("yt-dlp not installed - YouTube link skipped: %s", url)
            return None
        out_dir = self.download_dir / "youtube"
        out_dir.mkdir(exist_ok=True)
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "max_filesize": config.MAX_DOWNLOAD_MB * 1024 * 1024,
        }
        download = config.ENABLE_MEDIA
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=download)
            audio_path = Path(ydl.prepare_filename(info)) if download else None

        title = info.get("title") or url
        docs = []
        description = (info.get("description") or "").strip()
        if description:
            docs.append(Document(f"{title}\n\n{description}", {
                "source": title, "url": url, "source_type": "youtube", "content_type": "video_description"}))
        if audio_path and audio_path.exists():
            text = media.transcribe(audio_path)
            if text:
                docs.append(Document(text, {
                    "source": title, "url": url, "source_type": "youtube", "content_type": "video_transcript"}))
        return ScrapeResult(url, "youtube", docs)
