"""
DART session helper with retry, backoff, and randomized delays.
"""
from __future__ import annotations

import logging
import random
import time
from http.client import RemoteDisconnected
from typing import Optional

import requests
from bs4 import BeautifulSoup


class DartSession:
    def __init__(
        self,
        max_attempts: int = 5,
        min_delay: float = 1.2,
        max_delay: float = 2.5,
        backoff_base: float = 0.8,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.session = requests.Session()
        self.max_attempts = max_attempts
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.backoff_base = backoff_base
        self.logger = logger or logging.getLogger("DartSession")

        self.base_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "close",
        }

    def _sleep_between_requests(self) -> None:
        time.sleep(random.uniform(self.min_delay, self.max_delay))

    def _backoff_sleep(self, attempt: int) -> None:
        time.sleep(self.backoff_base * (2 ** (attempt - 1)))

    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        referer: Optional[str] = None,
        timeout: tuple[float, float] = (5, 20),
    ) -> Optional[requests.Response]:
        headers = dict(self.base_headers)
        if referer:
            headers["Referer"] = referer

        for attempt in range(1, self.max_attempts + 1):
            self._sleep_between_requests()
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=timeout)
                if response.status_code in {429, 500, 502, 503, 504}:
                    self.logger.warning(
                        "Retryable HTTP %s on %s (attempt %s/%s)",
                        response.status_code, url, attempt, self.max_attempts
                    )
                    self._backoff_sleep(attempt)
                    continue
                return response
            except RemoteDisconnected as exc:
                self.logger.warning(
                    "RemoteDisconnected on %s (attempt %s/%s): %s",
                    url, attempt, self.max_attempts, exc
                )
            except requests.exceptions.RequestException as exc:
                self.logger.warning(
                    "Request error on %s (attempt %s/%s): %s",
                    url, attempt, self.max_attempts, exc
                )

            self._backoff_sleep(attempt)

        self.logger.error("Skipping after %s failed attempts: %s", self.max_attempts, url)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    dart = DartSession(max_attempts=4)
    rcept_nos = ["20251224800811", "20251204000046"]

    for rcept_no in rcept_nos:
        url = "https://dart.fss.or.kr/dsaf001/main.do"
        resp = dart.get(url, params={"rcpNo": rcept_no}, referer="https://dart.fss.or.kr")
        if not resp or not resp.ok:
            continue

        html = resp.content.decode("euc-kr", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        title = soup.find("title")
        print(f"{rcept_no} -> {title.get_text(strip=True) if title else 'NO_TITLE'}")
