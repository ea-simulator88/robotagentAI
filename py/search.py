"""
Brave Search wrapper.

Brave Web Search API:
  - Endpoint: https://api.search.brave.com/res/v1/web/search
  - Header  : X-Subscription-Token: <BRAVE_API_KEY>
  - Free tier: $5 credit/month, ~2000 query → đủ dùng cho test
"""

from __future__ import annotations

import requests

from .common import BRAVE_API_KEY

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
TIMEOUT = 10

# Country codes Brave Search hỗ trợ. KHÔNG có VN.
# Ref: https://api.search.brave.com/app/documentation/web-search/codes
_BRAVE_COUNTRIES = {
    "ALL", "AR", "AU", "AT", "BE", "BR", "CA", "CL", "DK", "FI", "FR", "DE",
    "HK", "IN", "ID", "IT", "JP", "KR", "MY", "MX", "NL", "NZ", "NO", "CN",
    "PL", "PT", "PH", "RU", "SA", "ZA", "ES", "SE", "CH", "TW", "TR", "GB", "US",
}


def brave_search(
    query: str,
    count: int = 5,
    country: str = "ALL",
    search_lang: str = "vi",
) -> list[dict]:
    """Trả về list dict {title, url, description, age}.

    LƯU Ý: Brave country code KHÔNG có 'VN'. Danh sách hỗ trợ:
    ALL, AR, AU, AT, BE, BR, CA, CL, DK, FI, FR, DE, HK, IN, ID, IT, JP, KR,
    MY, MX, NL, NZ, NO, CN, PL, PT, PH, RU, SA, ZA, ES, SE, CH, TW, TR, GB, US.
    → mặc định 'ALL' để không bị 422 cho thị trường VN.
    """
    if not BRAVE_API_KEY:
        raise RuntimeError("BRAVE_API_KEY chưa set trong .env")
    if not query.strip():
        return []

    params = {
        "q": query,
        "count": count,
        "search_lang": search_lang,
        "safesearch": "moderate",
    }
    # Chỉ gửi country nếu hợp lệ — tránh 422 Unprocessable Entity
    if country and country.upper() in _BRAVE_COUNTRIES:
        params["country"] = country.upper()

    r = requests.get(
        BRAVE_URL,
        params=params,
        headers={
            "X-Subscription-Token": BRAVE_API_KEY,
            "Accept": "application/json",
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()

    results: list[dict] = []
    web = data.get("web") or {}
    for item in web.get("results") or []:
        results.append(
            {
                "title": (item.get("title") or "").strip(),
                "url": (item.get("url") or "").strip(),
                "description": (item.get("description") or "").strip(),
                "age": (item.get("age") or "").strip(),
            }
        )
    return results


def format_for_llm(results: list[dict], max_chars: int = 1800) -> str:
    """Định dạng kết quả cho LLM đọc. Cắt theo max_chars để không nổ context."""
    if not results:
        return "(không có kết quả)"
    out: list[str] = []
    total = 0
    for i, r in enumerate(results, start=1):
        age = f" ({r['age']})" if r.get("age") else ""
        snippet = (
            f"[{i}] {r['title']}{age}\n{r['description']}\nNguồn: {r['url']}"
        )
        if total + len(snippet) > max_chars and out:
            break
        out.append(snippet)
        total += len(snippet)
    return "\n\n".join(out)
