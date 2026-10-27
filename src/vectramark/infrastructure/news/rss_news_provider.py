"""RSS news retrieval with bounded network and parsing operations."""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from vectramark.core.exceptions import ResearchError

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
MAX_RESPONSE_BYTES = 2_000_000
USER_AGENT = "VectraMark/0.1 (market research RSS reader)"


@dataclass(frozen=True)
class NewsArticle:
    """Normalized news article from an RSS feed."""

    title: str
    url: str
    source: str | None
    published_at: datetime | None
    summary: str | None


class RSSNewsProvider:
    """Retrieve news from Google News RSS without an API key."""

    def __init__(self, timeout_seconds: float = 8.0) -> None:
        if not 1 <= timeout_seconds <= 30:
            raise ValueError("Timeout must be between 1 and 30 seconds.")
        self.timeout_seconds = timeout_seconds

    def fetch(self, query: str, limit: int = 20) -> list[NewsArticle]:
        """Fetch and normalize a bounded number of RSS articles."""

        clean_query = query.strip()
        if not clean_query or len(clean_query) > 300:
            raise ValueError("Query must contain between 1 and 300 characters.")
        if not 1 <= limit <= 30:
            raise ValueError("Limit must be between 1 and 30.")

        params = urlencode(
            {
                "q": clean_query,
                "hl": "en-US",
                "gl": "US",
                "ceid": "US:en",
            }
        )
        request = Request(
            f"{GOOGLE_NEWS_RSS_URL}?{params}",
            headers={"User-Agent": USER_AGENT},
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)

            if len(payload) > MAX_RESPONSE_BYTES:
                raise ResearchError("News feed exceeded the allowed response size.")

            root = ET.fromstring(payload)

        except ResearchError:
            raise
        except (URLError, TimeoutError, OSError) as exc:
            logger.warning("RSS provider request failed: %s", type(exc).__name__)
            raise ResearchError("News provider is temporarily unavailable.") from exc
        except ET.ParseError as exc:
            logger.warning("RSS provider returned invalid XML.")
            raise ResearchError("News provider returned an invalid feed.") from exc

        articles: list[NewsArticle] = []
        channel = root.find("channel")
        if channel is None:
            raise ResearchError("News provider returned an unsupported feed format.")

        for item in channel.findall("item"):
            title = (item.findtext("title") or "").strip()
            url = (item.findtext("link") or "").strip()

            if not title or not url.startswith(("https://", "http://")):
                continue

            source_text = item.findtext("source")
            source = source_text.strip() if source_text else None
            summary_text = item.findtext("description")
            summary = summary_text.strip() if summary_text else None
            published_text = item.findtext("pubDate")
            published_at = self._parse_date(published_text)

            articles.append(
                NewsArticle(
                    title=title[:500],
                    url=url[:2000],
                    source=source[:200] if source else None,
                    published_at=published_at,
                    summary=summary[:2000] if summary else None,
                )
            )

            if len(articles) >= limit:
                break

        return articles

    @staticmethod
    def _parse_date(value: str | None) -> datetime | None:
        """Parse common RSS dates; return None when the date is unavailable."""

        if not value or not value.strip():
            return None

        try:
            parsed = parsedate_to_datetime(value.strip())
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except (TypeError, ValueError, OverflowError):
            try:
                parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                return parsed.astimezone(UTC)
            except (TypeError, ValueError, OverflowError):
                return None
