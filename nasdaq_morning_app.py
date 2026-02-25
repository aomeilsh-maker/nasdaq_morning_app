#!/usr/bin/env python3
"""
NASDAQ-100 Morning Picker v2
- Daily run (intended via launchd 08:00)
- Multi-factor ranking + regime filter + news sentiment
- Outputs local HTML report and opens in browser

Disclaimer: Educational demo only, NOT financial advice.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import quote_plus

import pandas as pd
import requests
import yfinance as yf
from nasdaq_report_renderer import render_report


@dataclass
class Pick:
    symbol: str
    name: str
    score: float
    confidence: int
    grade: str
    price: float
    day_change: float
    ret_5d: float
    ret_20d: float
    ret_60d: float
    rsi14: float
    volume_ratio: float
    vol20_annual: float
    max_dd_60d: float
    cmf20: float
    obv_trend_20: float
    accumulation_tag: str
    signal_winrate_5d: float
    sentiment: str
    reason: str
    entry_hint: str
    stop_hint: str
    target_hint: str
    news: List[Dict[str, str]]


@dataclass
class WinrateSummary:
    total: int
    wins: int
    rate: float
    details: List[Dict[str, Any]]


@dataclass
class LongTermView:
    symbol: str
    name: str
    trend_label: str
    score: int
    price: float
    ret_1y: float
    ret_3y: float
    ma50: float
    ma200: float
    rsi14: float
    vol_1y: float
    max_dd_1y: float
    events: List[Dict[str, str]]
    analysis: str
    analysis_points: List[str]
    long_news: List[Dict[str, str]]
    news_pool_main: int
    news_pool_x: int
    news_used: int


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


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, math.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


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




def fetch_x_news_via_google_x_search(symbol: str, name: str, limit: int = 4) -> List[Dict[str, str]]:
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


def fetch_recent_news(symbol: str, name: str, limit: int = 6) -> tuple[List[Dict[str, str]], Dict[str, int]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    seen = set()

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
    candidates = _pull_google(f"{symbol} stock OR {name} when:3m", cap=max(limit * 4, 24))
    if len(candidates) < max(3, limit):
        candidates += _pull_google(f"{symbol} stock OR {name}", cap=max(limit * 3, 18))
    for it in candidates:
        link = it.get('url', '')
        if link and link not in seen:
            seen.add(link)
            mainstream_items.append(it)

    x_items: List[Dict[str, str]] = []
    for it in fetch_x_news_via_google_x_search(symbol, name, limit=max(6, limit * 2)):
        link = (it.get('url') or '').strip()
        if link and link not in seen:
            seen.add(link)
            x_items.append(it)

    all_items = mainstream_items + x_items
    if not all_items:
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
    for it in all_items:
        age = _age_days(str(it.get('pub_date', '')))
        freshness = 3.0 if age <= 1 else (2.0 if age <= 3 else (1.0 if age <= 7 else (0.5 if age <= 30 else 0.0)))
        source = str(it.get('source', ''))
        # X gets timeliness preference, mainstream gets reliability preference.
        source_weight = 1.4 if source == 'X' else 1.0
        score = freshness + source_weight
        scored.append((score, it))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Dynamic balancing by availability (not fixed counts)
    main_pool = [it for _, it in scored if it.get('source') != 'X']
    x_pool = [it for _, it in scored if it.get('source') == 'X']
    total_pool = max(1, len(main_pool) + len(x_pool))
    x_share = len(x_pool) / total_pool
    # adaptive target from pool share (bounded, variable)
    x_target = int(round(limit * x_share))
    x_target = max(0, min(len(x_pool), x_target))
    main_target = min(len(main_pool), limit - x_target)

    result: List[Dict[str, str]] = []
    result.extend(main_pool[:main_target])
    result.extend(x_pool[:x_target])

    # fill remaining by global score regardless of source
    if len(result) < limit:
        used = set(id(x) for x in result)
        for _, it in scored:
            if id(it) in used:
                continue
            result.append(it)
            used.add(id(it))
            if len(result) >= limit:
                break

    final_items = result[:limit]
    stats = {"main_pool": len(main_pool), "x_pool": len(x_pool), "used": len(final_items)}
    return final_items, stats


def sentiment_from_news(news: List[Dict[str, str]]) -> str:
    if not news:
        return "中性"
    pos = ["beat", "upgrade", "surge", "record", "strong", "growth", "buyback", "outperform"]
    neg = ["miss", "downgrade", "lawsuit", "probe", "drop", "weak", "cut", "recall"]
    score = 0
    for n in news:
        t = (n.get("title") or "").lower()
        score += sum(1 for w in pos if w in t)
        score -= sum(1 for w in neg if w in t)
    if score >= 2:
        return "偏利好"
    if score <= -2:
        return "偏利空"
    return "中性"


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


def _calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().fillna(0)
    signed = volume.where(direction >= 0, -volume)
    return signed.cumsum()


def _signal_winrate_5d(close: pd.Series) -> float:
    if len(close) < 140:
        return 0.0
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    rsi = compute_rsi(close, 14)
    future_5d = close.shift(-5) / close - 1.0

    mask = (close > ma20) & (ma20 > ma50) & (rsi >= 45) & (rsi <= 75)
    sample = future_5d[mask].dropna().tail(120)
    if len(sample) < 12:
        return 0.0
    return float((sample > 0).mean() * 100)


def score_symbol(symbol: str, name: str, risk_on: bool) -> Pick | None:
    try:
        hist = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False, threads=False)
        if hist is None or hist.empty or len(hist) < 80:
            return None

        close = _extract_series(hist, "Close", symbol).dropna()
        high = _extract_series(hist, "High", symbol).dropna()
        low = _extract_series(hist, "Low", symbol).dropna()
        vol = _extract_series(hist, "Volume", symbol).dropna()
        if len(close) < 90 or len(vol) < 30 or len(high) < 30 or len(low) < 30:
            return None

        price = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else price
        day_change = (price / prev - 1) * 100 if prev else 0
        ret_5d = (price / float(close.iloc[-6]) - 1) * 100
        ret_20d = (price / float(close.iloc[-21]) - 1) * 100
        ret_60d = (price / float(close.iloc[-61]) - 1) * 100

        ma20 = float(close.tail(20).mean())
        ma50 = float(close.tail(50).mean())
        rsi14 = float(compute_rsi(close, 14).iloc[-1])

        vol_recent = float(vol.tail(5).mean())
        vol_base = float(vol.tail(20).mean())
        volume_ratio = vol_recent / vol_base if vol_base else 1.0

        rets = close.pct_change().dropna()
        vol20_annual = float(rets.tail(20).std() * math.sqrt(252) * 100)
        roll_max = close.tail(60).cummax()
        dd = (close.tail(60) / roll_max - 1.0) * 100
        max_dd_60d = float(dd.min())

        mfm = ((close - low) - (high - close)) / (high - low).replace(0, math.nan)
        mfv = mfm.fillna(0) * vol
        cmf20 = float((mfv.tail(20).sum() / vol.tail(20).sum()) if vol.tail(20).sum() else 0.0)

        obv = _calc_obv(close, vol)
        obv_base = max(abs(float(obv.iloc[-21])) if len(obv) > 21 else 1.0, 1.0)
        obv_trend_20 = float((obv.iloc[-1] - obv.iloc[-21]) / obv_base * 100) if len(obv) > 21 else 0.0

        low60 = float(close.tail(60).min())
        dist_from_60d_low = (price / low60 - 1.0) * 100 if low60 > 0 else 0.0
        is_accumulating = (cmf20 > 0.05 and obv_trend_20 > 1.5 and 2.0 <= dist_from_60d_low <= 18.0)
        accumulation_tag = "疑似资金筑底" if is_accumulating else ("资金中性" if cmf20 >= 0 else "资金偏流出")

        signal_winrate_5d = _signal_winrate_5d(close)

        score = 0.0
        score += ret_20d * 0.25 + ret_60d * 0.20 + ret_5d * 0.25 + day_change * 0.10
        score += max(min((rsi14 - 45), 20), -20) * 0.35
        score += max(min(cmf20 * 40, 3.0), -3.0)
        score += max(min(obv_trend_20 * 0.35, 2.0), -2.0)
        score += max(min((signal_winrate_5d - 50) * 0.08, 2.0), -2.0)
        if price > ma20:
            score += 1.5
        if ma20 > ma50:
            score += 1.5
        if 1.0 <= volume_ratio <= 2.3:
            score += 1.0
        if is_accumulating:
            score += 1.5
        if vol20_annual > 65:
            score -= 1.2
        if max_dd_60d < -18:
            score -= 1.0
        if rsi14 > 80:
            score -= 1.5
        if not risk_on:
            score -= 2.0

        # Temporary raw confidence; final calibrated confidence is computed cross-sectionally in build_picks().
        conf = 0
        grade = "C"

        stop = price * (1 - max(0.03, min(0.09, vol20_annual / 100 * 0.6)))
        target = price * (1 + max(0.05, min(0.18, abs(max_dd_60d) / 100 * 0.8)))
        entry_hint = f"回踩 {ma20:.2f}~{price:.2f} 区间观察量价确认"
        stop_hint = f"{stop:.2f} 附近"
        target_hint = f"{target:.2f} 附近"

        reason_parts = []
        if ret_20d > 0:
            reason_parts.append(f"20日动量 {ret_20d:.1f}%")
        if ret_60d > 0:
            reason_parts.append(f"60日趋势 {ret_60d:.1f}%")
        if price > ma20:
            reason_parts.append("站上20日线")
        if ma20 > ma50:
            reason_parts.append("20日线高于50日线")
        if volume_ratio > 1.1:
            reason_parts.append(f"量能 {volume_ratio:.2f}x")
        reason_parts.append(f"CMF20 {cmf20:.2f}")
        reason_parts.append(f"OBV20变化 {obv_trend_20:.1f}%")
        if signal_winrate_5d > 0:
            reason_parts.append(f"历史同信号5日胜率 {signal_winrate_5d:.0f}%")
        reason_parts.append(accumulation_tag)
        reason = "；".join(reason_parts) if reason_parts else "信号一般"

        return Pick(
            symbol=symbol, name=name, score=score, confidence=conf, grade=grade,
            price=price, day_change=day_change, ret_5d=ret_5d, ret_20d=ret_20d, ret_60d=ret_60d,
            rsi14=rsi14, volume_ratio=volume_ratio, vol20_annual=vol20_annual, max_dd_60d=max_dd_60d,
            cmf20=cmf20, obv_trend_20=obv_trend_20, accumulation_tag=accumulation_tag, signal_winrate_5d=signal_winrate_5d,
            sentiment="中性", reason=reason, entry_hint=entry_hint, stop_hint=stop_hint, target_hint=target_hint,
            news=[]
        )
    except Exception:
        return None


def build_picks(limit: int = 5) -> tuple[list[Pick], dict]:
    table = get_nasdaq100_table()
    regime = market_regime()
    risk_on = bool(regime.get("risk_on", True))

    picks: List[Pick] = []
    for _, row in table.iterrows():
        p = score_symbol(row["symbol"], row["name"], risk_on=risk_on)
        if p:
            picks.append(p)

    picks.sort(key=lambda x: x.score, reverse=True)

    top = picks[:limit]

    # v2.2 confidence calibration:
    # Calibrate on the displayed shortlist to ensure practical differentiation (avoid all 100/all same).
    m = len(top)
    if m > 0:
        for idx, p in enumerate(top):
            rank_pct = 1.0 - (idx / max(m - 1, 1))  # top=1.0, last=0.0 on shortlist
            # Base confidence in [72, 92] for shortlist
            conf = int(round(72 + rank_pct * 20))
            # Small quality nudges (bounded)
            conf += int(max(min((p.signal_winrate_5d - 55) / 12, 2), -2))
            if p.accumulation_tag == "疑似资金筑底":
                conf += 2
            elif p.accumulation_tag == "资金偏流出":
                conf -= 2
            p.confidence = max(60, min(92, conf))
            p.grade = "A" if p.confidence >= 82 else ("B" if p.confidence >= 72 else "C")
    for p in top:
        p.news, _ = fetch_recent_news(p.symbol, p.name, limit=6)
        p.sentiment = sentiment_from_news(p.news)
    return top, regime


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []
    except Exception:
        return []


def _save_history(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def calc_last5_winrate_and_update_history(picks: List[Pick]) -> WinrateSummary:
    history_path = Path(__file__).with_name("nasdaq_reco_history.json")
    history = _load_history(history_path)

    today = dt.date.today().isoformat()
    symbols = [p.symbol for p in picks]

    # 先用历史中的前5个推荐日计算胜率（不含今天）
    prev_days = [r for r in history if r.get("date") != today]
    recent_days = prev_days[-5:]

    # 去重逻辑：在最近5个推荐日中，同一只股票只统计一次。
    # 统计口径使用该窗口内「最早被推荐那天」的 entry_price，与最新收盘价比较。
    earliest_entries: Dict[str, Dict[str, Any]] = {}
    for row in recent_days:
        day = str(row.get("date", ""))
        entries = row.get("picks", []) if isinstance(row.get("picks"), list) else []
        for e in entries:
            sym = str(e.get("symbol", "")).strip().upper()
            entry_price = float(e.get("entry_price", 0) or 0)
            if not sym or entry_price <= 0:
                continue
            # recent_days 按时间顺序（旧 -> 新），首次出现即为窗口内最早推荐
            if sym not in earliest_entries:
                earliest_entries[sym] = {"date": day, "entry_price": entry_price}

    all_symbols = sorted(earliest_entries.keys())
    current_close: Dict[str, float] = {}
    if all_symbols:
        try:
            data = yf.download(all_symbols, period="10d", interval="1d", auto_adjust=True, progress=False, threads=False)
            for s in all_symbols:
                c = _extract_series(data, "Close", s).dropna()
                if len(c):
                    current_close[s] = float(c.iloc[-1])
        except Exception:
            current_close = {}

    # 总胜率：按去重后的股票集合统计（每只股票只计一次）。
    total = 0
    wins = 0
    for sym in all_symbols:
        info = earliest_entries[sym]
        entry_price = float(info.get("entry_price", 0) or 0)
        now_price = current_close.get(sym)
        if entry_price <= 0 or now_price is None:
            continue
        total += 1
        if now_price > entry_price:
            wins += 1

    rate = round((wins / total * 100), 1) if total else 0.0

    # 按推荐日明细：使用“该推荐日的 entry_price”与“当前最新收盘价”比较。
    # 这里保留每个推荐日自己的口径，不按跨日去重。
    details: List[Dict[str, Any]] = []
    for row in recent_days:
        day = str(row.get("date", ""))
        day_total = 0
        day_wins = 0
        day_win_symbols: List[str] = []
        day_loss_symbols: List[str] = []
        entries = row.get("picks", []) if isinstance(row.get("picks"), list) else []

        for e in entries:
            sym = str(e.get("symbol", "")).strip().upper()
            entry_price = float(e.get("entry_price", 0) or 0)
            now_price = current_close.get(sym)
            if not sym or entry_price <= 0 or now_price is None:
                continue
            day_total += 1
            if now_price > entry_price:
                day_wins += 1
                day_win_symbols.append(sym)
            else:
                day_loss_symbols.append(sym)

        if day_total > 0:
            details.append({
                "date": day,
                "wins": day_wins,
                "total": day_total,
                "rate": round(day_wins / day_total * 100, 1),
                "win_symbols": day_win_symbols,
                "loss_symbols": day_loss_symbols,
            })

    # 写入/更新今天的推荐记录（用于未来5日统计）
    today_row = {
        "date": today,
        "symbols": symbols,
        "picks": [{"symbol": p.symbol, "entry_price": round(float(p.price), 4)} for p in picks],
        "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    history = [r for r in history if r.get("date") != today]
    history.append(today_row)
    history = history[-60:]  # 保留最近60次
    _save_history(history_path, history)

    return WinrateSummary(total=total, wins=wins, rate=rate, details=details)




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
        m = re.search(r"(20\d{2}-\d{1,2}-\d{1,2})", text)
        if m:
            return m.group(1)
        # Mar 18, 2026 / March 18, 2026
        m = re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*20\d{2}", text, re.I)
        if m:
            dtv = pd.to_datetime(m.group(0), errors='coerce')
            if pd.notna(dtv):
                return dtv.strftime('%Y-%m-%d')
    except Exception:
        pass
    return ''


def _symbol_event_queries(symbol: str, name: str) -> List[str]:
    base = [
        f"({symbol} OR {name}) (earnings date OR investor day OR conference OR product launch OR keynote OR guidance call OR hearing) when:3m",
    ]
    special = {
        'INTC': [
            'Intel Foundry Direct Connect 2026 March 24',
            'Intel FDC 2026 March 24',
            'Intel event March 2026 foundry',
            'site:intel.com Intel event March 2026',
        ],
        'NVDA': ['NVIDIA GTC 2026 date', 'site:nvidia.com event 2026'],
        'AMD': ['AMD Advancing AI 2026 event', 'site:amd.com events 2026'],
        'MSFT': ['Microsoft Build 2026 date', 'site:microsoft.com event 2026'],
        'AMZN': ['AWS re:Invent 2026 date', 'Amazon investor event 2026'],
        'META': ['Meta Connect 2026 date', 'Meta developer event 2026'],
        'GOOGL': ['Google I/O 2026 date', 'Alphabet investor event 2026'],
    }
    return base + special.get(symbol.upper(), [])


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
                    m = re.search(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}", low)
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

    # B) Unstructured future-event hints from mainstream news
    if len(items) < limit:
        extra = _future_events_from_news(symbol, name, limit=limit - len(items))
        for e in extra:
            t = str(e.get('title', '')).strip()
            if t and t not in seen:
                seen.add(t)
                items.append(e)
            if len(items) >= limit:
                break

    # C) loose fallback if still empty
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


def build_long_term_views() -> List[LongTermView]:
    targets = [
        ('INTC', 'Intel'),
        ('NVDA', 'NVIDIA'),
        ('AMD', 'AMD'),
        ('MSFT', 'Microsoft'),
        ('AMZN', 'Amazon'),
        ('META', 'Meta'),
        ('GOOGL', 'Google'),
    ]

    pos_words = ['beat', 'upgrade', 'surge', 'record', 'strong', 'growth', 'outperform', 'ai', 'partnership', 'buyback', 'expansion']
    neg_words = ['miss', 'downgrade', 'lawsuit', 'probe', 'drop', 'weak', 'cut', 'recall', 'antitrust', 'delay', 'ban']

    views: List[LongTermView] = []
    for symbol, name in targets:
        try:
            h = yf.download(symbol, period='5y', interval='1d', auto_adjust=True, progress=False, threads=False)
            close = _extract_series(h, 'Close', symbol).dropna()
            if len(close) < 260:
                continue

            price = float(close.iloc[-1])
            ret_1y = (price / float(close.iloc[-252]) - 1.0) * 100 if len(close) > 252 else 0.0
            ret_3y = (price / float(close.iloc[-756]) - 1.0) * 100 if len(close) > 756 else 0.0
            ma50 = float(close.tail(50).mean())
            ma200 = float(close.tail(200).mean())
            rsi14 = float(compute_rsi(close, 14).iloc[-1])
            vol_1y = float(close.pct_change().dropna().tail(252).std() * math.sqrt(252) * 100)
            roll_max = close.tail(252).cummax()
            dd = (close.tail(252) / roll_max - 1.0) * 100
            max_dd_1y = float(dd.min()) if len(dd) else 0.0

            events = _event_candidates_from_ticker(symbol, name, limit=6)
            # 动态新闻条数：不再写死8条，按可用活动信息和个股复杂度自适应
            news_limit = min(14, max(6, len(events) + 6))
            news_items, news_stats = fetch_recent_news(symbol, name, limit=news_limit)
            pos_cnt = 0
            neg_cnt = 0
            pos_w = 0.0
            neg_w = 0.0
            themes = set()
            mainstream_n = 0
            x_n = 0
            for e in news_items:
                title = (e.get('title') or '').lower()
                source = (e.get('source') or '').lower()
                weight = 0.4 if source == 'x' else 1.0  # X信号纳入，但权重更低
                if source == 'x':
                    x_n += 1
                else:
                    mainstream_n += 1

                p_hits_n = sum(1 for w in pos_words if w in title)
                n_hits_n = sum(1 for w in neg_words if w in title)
                pos_cnt += p_hits_n
                neg_cnt += n_hits_n
                pos_w += p_hits_n * weight
                neg_w += n_hits_n * weight

                if 'earnings' in title or 'guidance' in title:
                    themes.add('财报/指引')
                if 'ai' in title or 'chip' in title or 'gpu' in title:
                    themes.add('AI/芯片')
                if 'cloud' in title or 'data center' in title:
                    themes.add('云/数据中心')
                if 'regulation' in title or 'antitrust' in title or 'probe' in title:
                    themes.add('监管/反垄断')
                if 'buyback' in title or 'dividend' in title:
                    themes.add('资本回报')

            msg_bias = pos_cnt - neg_cnt
            msg_bias_w = pos_w - neg_w

            score = 52
            score += 10 if price > ma200 else -8
            score += 8 if ma50 > ma200 else -6
            score += max(min(ret_1y * 0.12, 10), -10)
            score += max(min(ret_3y * 0.05, 8), -8)
            score += max(min(msg_bias_w * 2.8, 16), -16)  # 用加权消息面驱动长线判断
            if 45 <= rsi14 <= 72:
                score += 4
            score += max(min((20 - abs(max_dd_1y)) * 0.2, 3), -5)
            score = int(max(0, min(100, round(score))))

            if score >= 72:
                trend_label = '长线上行（消息面+趋势共振）'
            elif score >= 58:
                trend_label = '长线偏多（消息面中性偏利好）'
            elif score >= 45:
                trend_label = '长线中性（消息面分歧）'
            else:
                trend_label = '长线偏弱（消息/趋势承压）'

            if msg_bias_w >= 2.5:
                msg_text = '近期消息面明显偏利好'
            elif msg_bias_w >= 0.8:
                msg_text = '近期消息面小幅偏利好'
            elif msg_bias_w <= -2.5:
                msg_text = '近期消息面明显偏利空'
            elif msg_bias_w <= -0.8:
                msg_text = '近期消息面小幅偏利空'
            else:
                msg_text = '近期消息面中性'

            theme_text = '、'.join(sorted(themes)) if themes else '暂无明确主题'

            pos_events: List[str] = []
            neg_events: List[str] = []
            for e in news_items:
                t = (e.get('title') or '').strip()
                low = t.lower()
                src = e.get('source', '')
                src_tag = f"{src}(低权重)" if str(src).lower() == 'x' else str(src)
                p_hits = [w for w in pos_words if w in low]
                n_hits = [w for w in neg_words if w in low]
                if p_hits and len(pos_events) < 2:
                    pos_events.append(f"{t}（来源: {src_tag}，触发: {','.join(p_hits[:2])}）")
                if n_hits and len(neg_events) < 2:
                    neg_events.append(f"{t}（来源: {src_tag}，触发: {','.join(n_hits[:2])}）")

            pos_text = '；'.join(pos_events) if pos_events else '暂无显著利好事件命中'
            neg_text = '；'.join(neg_events) if neg_events else '暂无显著利空事件命中'

            analysis = (
                f"{name}：{msg_text}，综合判断：{trend_label}。"
            )
            analysis_points = [
                f"消息面结论：{msg_text}（原始利好词{pos_cnt}/利空词{neg_cnt}；加权后{pos_w:.1f}/{neg_w:.1f}）",
                f"信号来源结构：主流/公告 {mainstream_n} 条，X {x_n} 条（X按0.4权重计入）",
                f"消息池透明度：主流池 {news_stats.get('main_pool',0)} 条，X池 {news_stats.get('x_pool',0)} 条，最终采用 {news_stats.get('used',0)} 条",
                f"事件主题：{theme_text}",
                f"利好依据：{pos_text}",
                f"利空核对：{neg_text}",
                f"技术面辅助：价格{'高于' if price > ma200 else '低于'}200日线，1年收益{ret_1y:.1f}%，3年收益{ret_3y:.1f}%",
            ]

            views.append(LongTermView(
                symbol=symbol, name=name, trend_label=trend_label, score=score, price=price,
                ret_1y=ret_1y, ret_3y=ret_3y, ma50=ma50, ma200=ma200, rsi14=rsi14,
                vol_1y=vol_1y, max_dd_1y=max_dd_1y, events=events, analysis=analysis, analysis_points=analysis_points,
                long_news=[{"source": str(n.get("source", "")), "title": str(n.get("title", "")), "url": str(n.get("url", ""))} for n in news_items],
                news_pool_main=int(news_stats.get("main_pool", 0)), news_pool_x=int(news_stats.get("x_pool", 0)), news_used=int(news_stats.get("used", len(news_items)))
            ))
        except Exception:
            continue

    views.sort(key=lambda x: x.score, reverse=True)
    return views


def main() -> None:
    picks, regime = build_picks(limit=10)
    winrate = calc_last5_winrate_and_update_history(picks)
    long_views = build_long_term_views()
    x_status = get_x_fetch_status("NVDA")
    render_report(picks, regime, winrate, long_views, x_status)


if __name__ == "__main__":
    main()
