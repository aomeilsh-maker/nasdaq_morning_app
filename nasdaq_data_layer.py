from __future__ import annotations

import datetime as dt
import io
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List
from urllib.parse import quote_plus

import pandas as pd
import requests
import yfinance as yf


# News pipeline knobs (single-source-of-truth; avoid scattered magic numbers)
NEWS_OUTPUT_LIMIT = 6
NEWS_FETCH_POOL_MAIN_PRIMARY = 36
NEWS_FETCH_POOL_MAIN_FALLBACK = 30
NEWS_FETCH_POOL_X = 20
# Relevance/quality thresholds (applies to both long-term and short-term news)
NEWS_MIN_RELEVANCE_SCORE = 5
NEWS_MIN_FINAL_SCORE = 8.5


def get_nasdaq100_table() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if "ticker" in cols or "ticker symbol" in cols:
                table = t.copy()
                break
        else:
            raise RuntimeError("Constituents table not found")

        colmap = {c: str(c).strip().lower() for c in table.columns}
        ticker_col = next(c for c, lc in colmap.items() if lc in ("ticker", "ticker symbol"))
        company_col = next((c for c, lc in colmap.items() if lc in ("company", "company name")), None)

        out = pd.DataFrame()
        out["symbol"] = table[ticker_col].astype(str).str.replace(".", "-", regex=False).str.strip()
        out["name"] = table[company_col].astype(str).str.strip() if company_col else out["symbol"]
        out = out.dropna().drop_duplicates(subset=["symbol"]).reset_index(drop=True)
        if not out.empty:
            return out
    except Exception:
        pass

    fallback_symbols = [
        "AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN","AMZN","ANSS","APP","ARM","ASML","AVGO","AXON","AZN","BIIB","BKNG",
        "CDNS","CEG","CHTR","CMCSA","COST","CPRT","CRWD","CSCO","CSX","CTAS","CTSH","DASH","DDOG","DXCM","EA","EXC","FAST","FTNT","GEHC","GFS",
        "GILD","GOOG","GOOGL","HON","IDXX","ILMN","INTC","INTU","ISRG","KDP","KHC","KLAC","LIN","LRCX","LULU","MAR","MCHP","MDLZ","MELI","META",
        "MNST","MRNA","MRVL","MSFT","MSTR","MU","NFLX","NVDA","NXPI","ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD","PEP","PYPL","QCOM","REGN",
        "ROP","ROST","SBUX","SNPS","TEAM","TMUS","TSLA","TTD","TTWO","TXN","VRSK","VRTX","WBD","WDAY","XEL","ZS",
    ]
    return pd.DataFrame({"symbol": fallback_symbols, "name": fallback_symbols})

def _extract_series(df: pd.DataFrame, field: str, symbol: str) -> pd.Series:
    if field in df.columns:
        s = df[field]
        if isinstance(s, pd.DataFrame):
            if symbol in s.columns:
                return s[symbol]
            return s.iloc[:, 0]
        return s
    if isinstance(df.columns, pd.MultiIndex):
        candidates = [c for c in df.columns if str(c[0]).lower() == field.lower()]
        if candidates:
            for c in candidates:
                if len(c) > 1 and str(c[1]).upper() == symbol.upper():
                    return df[c]
            return df[candidates[0]]
    return pd.Series(dtype=float)

def fetch_x_news_via_google_x_search(symbol: str, name: str, limit: int = 6) -> List[Dict[str, str]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    items: List[Dict[str, str]] = []
    seen = set()
    try:
        q = quote_plus(f"site:x.com ({symbol} OR {name}) when:3m")
        rss = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        r = requests.get(rss, headers=headers, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for it in root.findall("./channel/item")[: max(limit * 2, 10)]:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            low = link.lower()
            if not title or not link:
                continue
            if ('x.com/' not in low and 'twitter.com/' not in low and 'news.google.com' not in low):
                continue
            if link in seen:
                continue
            seen.add(link)
            pub = (it.findtext("pubDate") or "").strip(); items.append({"source": "X", "title": title, "url": link, "pub_date": pub})
            if len(items) >= limit:
                break
    except Exception:
        pass
    return items[:limit]

def fetch_recent_news(symbol: str, name: str, limit: int | None = NEWS_OUTPUT_LIMIT) -> tuple[List[Dict[str, str]], Dict[str, int]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    seen = set()
    # limit=None means "return all filtered/scored items" (used by long-term section).
    effective_limit = limit if (isinstance(limit, int) and limit > 0) else None

    def _pull_google(query: str, cap: int = 18) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        try:
            rss = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
            r = requests.get(rss, headers=headers, timeout=15)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            for it in root.findall("./channel/item")[:cap]:
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                pub = (it.findtext("pubDate") or "").strip()
                if title and link:
                    out.append({"source": "Google News", "title": title, "url": link, "pub_date": pub})
        except Exception:
            pass
        return out

    mainstream_items: List[Dict[str, str]] = []
    # 先尽量拉全候选池，再做关键词相关性筛选（而不是“抓几条就用几条”）。
    candidates = _pull_google(f"{symbol} stock OR {name} when:3m", cap=NEWS_FETCH_POOL_MAIN_PRIMARY)
    min_needed = max(8, (effective_limit or NEWS_OUTPUT_LIMIT) * 2)
    if len(candidates) < min_needed:
        candidates += _pull_google(f"{symbol} stock OR {name}", cap=NEWS_FETCH_POOL_MAIN_FALLBACK)
    for it in candidates:
        link = it.get('url', '')
        if link and link not in seen:
            seen.add(link)
            mainstream_items.append(it)

    x_items: List[Dict[str, str]] = []
    # X 侧同样先扩大候选池，再由关键词过滤 + 评分挑选。
    x_fetch_limit = NEWS_FETCH_POOL_X
    for it in fetch_x_news_via_google_x_search(symbol, name, limit=x_fetch_limit):
        link = (it.get('url') or '').strip()
        if link and link not in seen:
            seen.add(link)
            x_items.append(it)

    all_items = mainstream_items + x_items
    if not all_items:
        return [], {"main_pool": 0, "x_pool": 0, "used": 0}

    symbol_upper = symbol.upper().strip()
    name_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", name) if len(t) >= 3]
    topic_keywords = {
        "stock", "shares", "price", "rating", "upgrade", "downgrade", "earnings", "revenue",
        "guidance", "forecast", "outlook", "ai", "chip", "semiconductor", "data center", "cloud",
        "launch", "product", "conference", "investor", "buyback", "acquisition", "merger",
    }

    def _keyword_relevance(it: Dict[str, str]) -> int:
        title = str(it.get("title", ""))
        low = title.lower()
        score = 0
        if symbol_upper and symbol_upper in title.upper():
            score += 3
        if any(tok in low for tok in name_tokens):
            score += 2
        if any(k in low for k in topic_keywords):
            score += 1
        return score

    # 相关性过滤：仅保留达到最低相关性阈值的消息。
    # 不再回退到“全量不过滤”，避免低质量/弱相关新闻进入输出。
    filtered_items = [it for it in all_items if _keyword_relevance(it) >= NEWS_MIN_RELEVANCE_SCORE]
    if not filtered_items:
        return [], {"main_pool": 0, "x_pool": 0, "used": 0}

    def _age_days(pub: str) -> float:
        try:
            if not pub:
                return 999
            dtv = pd.to_datetime(pub, errors='coerce')
            if pd.isna(dtv):
                return 999
            if getattr(dtv, 'tzinfo', None) is not None:
                dtv = dtv.tz_convert(None)
            now = pd.Timestamp.now()
            return max(0.0, (now - dtv).total_seconds() / 86400.0)
        except Exception:
            return 999

    scored = []
    for it in filtered_items:
        relevance = _keyword_relevance(it)
        age = _age_days(str(it.get('pub_date', '')))
        freshness = 3.0 if age <= 1 else (2.0 if age <= 3 else (1.0 if age <= 7 else (0.5 if age <= 30 else 0.0)))
        source = str(it.get('source', ''))
        # X偏时效、主流偏可靠：保留轻微来源权重，但让相关性主导。
        source_weight = 1.4 if source == 'X' else 1.0
        final_score = relevance * 1.2 + freshness + source_weight
        if final_score < NEWS_MIN_FINAL_SCORE:
            continue
        scored.append((final_score, it))

    if not scored:
        return [], {"main_pool": 0, "x_pool": 0, "used": 0}

    scored.sort(key=lambda x: x[0], reverse=True)

    # Dynamic balancing by availability (not fixed counts)
    main_pool = [it for _, it in scored if it.get('source') != 'X']
    x_pool = [it for _, it in scored if it.get('source') == 'X']

    if effective_limit is None:
        # Return all filtered+scored items (for long-term analysis transparency).
        final_items = [it for _, it in scored]
    else:
        total_pool = max(1, len(main_pool) + len(x_pool))
        x_share = len(x_pool) / total_pool
        # adaptive target from pool share (bounded, variable)
        x_target = int(round(effective_limit * x_share))
        x_target = max(0, min(len(x_pool), x_target))
        main_target = min(len(main_pool), effective_limit - x_target)

        result: List[Dict[str, str]] = []
        result.extend(main_pool[:main_target])
        result.extend(x_pool[:x_target])

        # fill remaining by global score regardless of source
        if len(result) < effective_limit:
            used = set(id(x) for x in result)
            for _, it in scored:
                if id(it) in used:
                    continue
                result.append(it)
                used.add(id(it))
                if len(result) >= effective_limit:
                    break

        final_items = result[:effective_limit]
    stats = {"main_pool": len(main_pool), "x_pool": len(x_pool), "used": len(final_items)}
    return final_items, stats

def market_regime() -> Dict[str, float | bool]:
    try:
        h = yf.download("QQQ", period="1y", interval="1d", auto_adjust=True, progress=False, threads=False)
        close = _extract_series(h, "Close", "QQQ").dropna()
        if len(close) < 210:
            return {"risk_on": True, "price": float(close.iloc[-1]) if len(close) else 0.0, "ma50": 0.0, "ma200": 0.0}
        price = float(close.iloc[-1])
        ma50 = float(close.tail(50).mean())
        ma200 = float(close.tail(200).mean())
        return {"risk_on": price > ma50 and ma50 > ma200, "price": price, "ma50": ma50, "ma200": ma200}
    except Exception:
        return {"risk_on": True, "price": 0.0, "ma50": 0.0, "ma200": 0.0}

def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if hasattr(v, 'item'):
            return float(v.item())
        return float(v)
    except Exception:
        return default

def _extract_future_date_from_text(text: str) -> str:
    try:
        m = re.search(r"\b(20\d{2}-\d{1,2}-\d{1,2})\b", text)
        if m:
            return m.group(1)
        # Mar 18, 2026 / March 18, 2026
        m = re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*20\d{2}\b", text, re.I)
        if m:
            dtv = pd.to_datetime(m.group(0), errors='coerce')
            if pd.notna(dtv):
                return dtv.strftime('%Y-%m-%d')
    except Exception:
        pass
    return ''

def _symbol_event_queries(symbol: str, name: str) -> List[str]:
    """Build generic future-event search queries (no hard-coded ticker events)."""
    return [
        f"({symbol} OR {name}) (earnings date OR investor day OR conference OR product launch OR keynote OR guidance call OR hearing) when:3m",
    ]


# Official company feeds for direct (non-Google) event/news lookup.
# Keep this explicit so long-term event detection can use first-party sources.
OFFICIAL_EVENT_FEEDS: Dict[str, List[str]] = {
    "INTC": [
        "https://www.intc.com/rss/news-releases.xml",
    ],
    "NVDA": [
        "https://nvidianews.nvidia.com/releases.xml",
    ],
    "AMD": [
        "https://ir.amd.com/rss/news-releases.xml",
    ],
    "MSFT": [
        "https://news.microsoft.com/feed/",
        "https://blogs.microsoft.com/feed/",
    ],
    "AMZN": [
        "https://www.aboutamazon.com/news.xml",
        "https://www.amazon.com/gp/help/rss/about",
    ],
    "META": [
        "https://about.fb.com/news/feed/",
    ],
    "GOOGL": [
        "https://blog.google/rss/",
        "https://www.blog.google/rss/",
    ],
}


def _future_events_from_official_sites(symbol: str, name: str, limit: int = 3) -> List[Dict[str, str]]:
    """Pull event hints directly from company official RSS/Atom feeds."""
    feeds = OFFICIAL_EVENT_FEEDS.get(symbol.upper(), [])
    if not feeds:
        return []

    headers = {"User-Agent": "Mozilla/5.0"}
    out: List[Dict[str, str]] = []
    seen = set()
    now = pd.Timestamp.now().normalize()
    horizon = now + pd.Timedelta(days=120)
    event_keywords = [
        'earnings', 'investor', 'conference', 'launch', 'keynote', 'event',
        'webcast', 'summit', 'connect', 'build', 'gtc', 're:invent', 'foundry',
    ]

    for feed in feeds:
        if len(out) >= limit:
            break
        try:
            resp = requests.get(feed, headers=headers, timeout=15)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            # RSS item + Atom entry support
            entries = root.findall('.//item') + root.findall('.//{http://www.w3.org/2005/Atom}entry')

            for it in entries[:40]:
                if len(out) >= limit:
                    break

                title = (it.findtext('title') or it.findtext('{http://www.w3.org/2005/Atom}title') or '').strip()
                link = (it.findtext('link') or '').strip()
                if not link:
                    link_node = it.find('{http://www.w3.org/2005/Atom}link')
                    if link_node is not None:
                        link = (link_node.attrib.get('href') or '').strip()
                pub = (
                    it.findtext('pubDate')
                    or it.findtext('{http://www.w3.org/2005/Atom}updated')
                    or it.findtext('{http://www.w3.org/2005/Atom}published')
                    or ''
                ).strip()

                if not title or not link or link in seen:
                    continue
                low = title.lower()
                if not any(k in low for k in event_keywords):
                    continue

                # Prefer explicit future date in title/link; fallback to recent official event hints.
                d = _extract_future_date_from_text(f"{title} {link}")
                dtv = pd.to_datetime(d, errors='coerce') if d else pd.NaT
                if pd.notna(dtv) and not (now <= dtv.normalize() <= horizon):
                    continue

                if pd.isna(dtv) and pub:
                    try:
                        pubdt = pd.to_datetime(pub, errors='coerce')
                        if pd.notna(pubdt):
                            if pubdt.tzinfo is not None:
                                pubdt = pubdt.tz_convert(None)
                            if pubdt.normalize() < (now - pd.Timedelta(days=45)):
                                continue
                    except Exception:
                        pass

                seen.add(link)
                date_text = f"（预计日期: {dtv.strftime('%Y-%m-%d')}）" if pd.notna(dtv) else "（日期待确认）"
                out.append({
                    'type': '未来活动',
                    'title': f"{title}{date_text}",
                    'source': 'Official site feed',
                    'url': link,
                })
        except Exception:
            continue

    return out[:limit]

def _future_events_from_news(symbol: str, name: str, limit: int = 4) -> List[Dict[str, str]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    out: List[Dict[str, str]] = []
    seen = set()

    queries = _symbol_event_queries(symbol, name)
    now = pd.Timestamp.now().normalize()
    horizon = now + pd.Timedelta(days=120)  # give buffer to avoid missing just-over-3m announced events

    for qraw in queries:
        if len(out) >= limit:
            break
        q = quote_plus(qraw)
        try:
            rss = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
            r = requests.get(rss, headers=headers, timeout=15)
            r.raise_for_status()
            root = ET.fromstring(r.text)

            for it in root.findall("./channel/item")[:30]:
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                pub = (it.findtext("pubDate") or "").strip()
                if not title or not link or link in seen:
                    continue

                low = title.lower()
                if not any(k in low for k in ['earnings', 'investor day', 'conference', 'launch', 'keynote', 'guidance', 'hearing', 'event', 'call', 'fdc', 'foundry', 'connect', 'build', 'gtc', 'i/o']):
                    continue

                d = _extract_future_date_from_text(title + ' ' + link)
                dtv = pd.to_datetime(d, errors='coerce') if d else pd.NaT

                # fallback parse patterns like "March 24" without year -> assume this/next year
                if pd.isna(dtv):
                    m = re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b", low)
                    if m:
                        guess = f"{m.group(0)} {now.year}"
                        g = pd.to_datetime(guess, errors='coerce')
                        if pd.notna(g) and g.normalize() < now:
                            g = pd.to_datetime(f"{m.group(0)} {now.year+1}", errors='coerce')
                        dtv = g

                if pd.isna(dtv):
                    continue
                if not (now <= dtv.normalize() <= horizon):
                    continue

                if pub:
                    try:
                        pubdt = pd.to_datetime(pub, errors='coerce')
                        if pd.notna(pubdt):
                            if pubdt.tzinfo is not None:
                                pubdt = pubdt.tz_convert(None)
                            if pubdt.normalize() < (now - pd.Timedelta(days=90)):
                                continue
                    except Exception:
                        pass

                seen.add(link)
                out.append({
                    'type': '未来活动',
                    'title': f"{title}（预计日期: {dtv.strftime('%Y-%m-%d')}）",
                    'source': 'Google News future-event',
                    'url': link,
                })
                if len(out) >= limit:
                    break
        except Exception:
            continue

    return out

def _future_events_from_news_loose(symbol: str, name: str, limit: int = 3) -> List[Dict[str, str]]:
    """Fallback upcoming-activity hints when explicit date is missing."""
    headers = {"User-Agent": "Mozilla/5.0"}
    out: List[Dict[str, str]] = []
    seen = set()
    q = quote_plus(f"({symbol} OR {name}) (will OR expected OR scheduled OR to announce OR upcoming) (earnings OR investor day OR conference OR launch OR keynote OR hearing) when:3m")
    try:
        rss = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        r = requests.get(rss, headers=headers, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        now = pd.Timestamp.now().normalize()
        for it in root.findall("./channel/item")[:20]:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            pub = (it.findtext("pubDate") or "").strip()
            if not title or not link or link in seen:
                continue
            low = title.lower()
            if not any(k in low for k in ['earnings', 'investor day', 'conference', 'launch', 'keynote', 'hearing', 'guidance', 'call']):
                continue
            # keep only recent posts to reduce stale quarter recaps
            if pub:
                try:
                    pubdt = pd.to_datetime(pub, errors='coerce')
                    if pd.notna(pubdt):
                        if pubdt.tzinfo is not None:
                            pubdt = pubdt.tz_convert(None)
                        if pubdt.normalize() < (now - pd.Timedelta(days=35)):
                            continue
                except Exception:
                    pass
            seen.add(link)
            out.append({
                'type': '未来活动',
                'title': f"{title}（日期待确认）",
                'source': 'Google News future-event(loose)',
                'url': link,
            })
            if len(out) >= limit:
                break
    except Exception:
        pass
    return out

def _event_candidates_from_ticker(symbol: str, name: str, limit: int = 5) -> List[Dict[str, str]]:
    """重大事件：仅保留未发生的未来活动（未来3个月）。"""
    items: List[Dict[str, str]] = []
    seen = set()

    now = pd.Timestamp.now()
    horizon = now + pd.Timedelta(days=92)

    # A) Structured calendar events
    try:
        t = yf.Ticker(symbol)
        cal = t.calendar
        if cal is not None and not getattr(cal, 'empty', True):
            for idx in cal.index:
                label = str(idx)
                val = cal.loc[idx].iloc[0] if hasattr(cal.loc[idx], 'iloc') else cal.loc[idx]
                txt = str(val)
                if not txt or txt == 'NaT':
                    continue
                keep = False
                dt_txt = txt
                try:
                    dtv = pd.to_datetime(txt, errors='coerce')
                    if pd.notna(dtv):
                        if getattr(dtv, 'tzinfo', None):
                            now_tz = pd.Timestamp.now(tz=dtv.tz)
                            horizon_tz = now_tz + pd.Timedelta(days=92)
                            keep = (dtv >= now_tz) and (dtv <= horizon_tz)
                        else:
                            keep = (dtv >= now) and (dtv <= horizon)
                        dt_txt = dtv.strftime('%Y-%m-%d %H:%M')
                except Exception:
                    keep = False
                if keep:
                    title = f"{label}: {dt_txt}"
                    if title not in seen:
                        seen.add(title)
                        items.append({'type': '未来活动', 'title': title, 'source': 'yfinance calendar'})
                if len(items) >= limit:
                    break

        # A2) earnings dates endpoint fallback
        if len(items) < limit:
            try:
                ed = t.get_earnings_dates(limit=8)
                if ed is not None and not ed.empty:
                    for idx, _ in ed.iterrows():
                        dtv = pd.to_datetime(idx, errors='coerce')
                        if pd.isna(dtv):
                            continue
                        dt_naive = dtv.tz_convert(None) if getattr(dtv, 'tzinfo', None) is not None else dtv
                        if now <= dt_naive <= horizon:
                            title = f"Earnings Date: {dt_naive.strftime('%Y-%m-%d')}"
                            if title not in seen:
                                seen.add(title)
                                items.append({'type': '未来活动', 'title': title, 'source': 'yfinance earnings_dates'})
                        if len(items) >= limit:
                            break
            except Exception:
                pass
    except Exception:
        pass

    # B) First-party company websites (official feeds)
    if len(items) < limit:
        official = _future_events_from_official_sites(symbol, name, limit=limit - len(items))
        for e in official:
            t = str(e.get('title', '')).strip()
            if t and t not in seen:
                seen.add(t)
                items.append(e)
            if len(items) >= limit:
                break

    # C) Unstructured future-event hints from mainstream news
    if len(items) < limit:
        extra = _future_events_from_news(symbol, name, limit=limit - len(items))
        for e in extra:
            t = str(e.get('title', '')).strip()
            if t and t not in seen:
                seen.add(t)
                items.append(e)
            if len(items) >= limit:
                break

    # D) loose fallback if still empty
    if len(items) < limit:
        extra2 = _future_events_from_news_loose(symbol, name, limit=limit - len(items))
        for e in extra2:
            t = str(e.get('title', '')).strip()
            if t and t not in seen:
                seen.add(t)
                items.append(e)
            if len(items) >= limit:
                break

    return items[:limit]

def get_x_fetch_status(sample_symbol: str = "NVDA") -> Dict[str, object]:
    sample = fetch_x_news_via_google_x_search(sample_symbol, sample_symbol, limit=20)
    return {
        "ok": 1 if len(sample) > 0 else 0,
        "total": 1,
        "sample_count": len(sample),
        "ok_hosts": ["google-site-x"] if len(sample) > 0 else [],
        "failed_hosts": [] if len(sample) > 0 else ["google-site-x(empty)"],
        "fallback_count": len(sample),
        "mode": "web-site-x",
    }

