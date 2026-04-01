from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional


def http_json(
    method: str,
    url: str,
    body: Any = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 45.0,
) -> Any:
    data = None
    h = {"Accept": "application/json", "User-Agent": "passive-liquidity-bot/1.0"}
    if headers:
        h.update(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}: {err}") from e
