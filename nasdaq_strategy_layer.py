from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

from nasdaq_data_layer import (
    NEWS_OUTPUT_LIMIT,
    _event_candidates_from_ticker,
    _extract_series,
    _safe_float,
    fetch_recent_news,
    fetch_x_news_via_google_x_search,
    get_nasdaq100_table,
    market_regime,
)


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
    # Subset metric: symbols with estimated 5D winrate >= threshold at least once in last 5 reco days.
    high_est_total: int
    high_est_wins: int
    high_est_rate: float
    high_est_threshold: float

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

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, math.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

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

def score_symbol(symbol: str, name: str, risk_on: bool, hist: pd.DataFrame | None = None) -> Pick | None:
    try:
        # Allow preloaded history injection (batch download in build_picks) to reduce request count.
        if hist is None:
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

        # Keep short-term rationale schema fixed across symbols (avoid cards feeling inconsistent).
        trend_text = f"均线结构: 价{'上' if price > ma20 else '下'}20日线 / 20日线{'上' if ma20 > ma50 else '下'}50日线"
        volume_text = f"量能: {volume_ratio:.2f}x"
        flow_text = f"资金流: CMF20 {cmf20:.2f}, OBV20 {obv_trend_20:.1f}%（{accumulation_tag}）"
        risk_text = f"风险: 20D波动 {vol20_annual:.1f}%, 60D回撤 {max_dd_60d:.1f}%"
        win_text = f"同信号5日胜率: {signal_winrate_5d:.0f}%" if signal_winrate_5d > 0 else "同信号5日胜率: 样本不足"
        reason = (
            f"动量: 5日 {ret_5d:.1f}% / 20日 {ret_20d:.1f}% / 60日 {ret_60d:.1f}%；"
            f"{trend_text}；{volume_text}；{flow_text}；{risk_text}；{win_text}"
        )

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

    # Batch-download reduces network overhead vs per-symbol downloads (~100 -> 1 request).
    symbols = [str(s).strip().upper() for s in table["symbol"].tolist() if str(s).strip()]
    try:
        hist_all = yf.download(symbols, period="1y", interval="1d", auto_adjust=True, progress=False, threads=False)
    except Exception:
        hist_all = pd.DataFrame()

    for _, row in table.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        name = str(row["name"]).strip()

        # With a batched DataFrame, _extract_series can pull per-symbol OHLCV slices.
        p = score_symbol(symbol, name, risk_on=risk_on, hist=hist_all)
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
        # 短线新闻同样按阈值过滤后全量输出，不再固定6条。
        p.news, _ = fetch_recent_news(p.symbol, p.name, limit=None)
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


def _backfill_est_winrate_5d(history: list[dict]) -> bool:
    """Backfill missing est_winrate_5d for historical picks using each recommendation day's market context.

    For each history row(date), compute signal winrate as-of that day (using data ending on that day),
    then write est_winrate_5d into picks entries if missing.
    """
    changed = False
    for row in history:
        day = str(row.get("date", "")).strip()
        entries = row.get("picks", []) if isinstance(row.get("picks"), list) else []
        if not day or not entries:
            continue

        missing_symbols: list[str] = []
        for e in entries:
            if "est_winrate_5d" not in e:
                sym = str(e.get("symbol", "")).strip().upper()
                if sym:
                    missing_symbols.append(sym)

        if not missing_symbols:
            continue

        try:
            asof = pd.Timestamp(day)
            start = (asof - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
            end = (asof + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            data = yf.download(sorted(set(missing_symbols)), start=start, end=end, interval="1d", auto_adjust=True, progress=False, threads=False)

            est_map: Dict[str, float] = {}
            for sym in sorted(set(missing_symbols)):
                close = _extract_series(data, "Close", sym).dropna()
                est_map[sym] = float(_signal_winrate_5d(close)) if len(close) else 0.0

            for e in entries:
                if "est_winrate_5d" in e:
                    continue
                sym = str(e.get("symbol", "")).strip().upper()
                if not sym:
                    continue
                e["est_winrate_5d"] = round(float(est_map.get(sym, 0.0)), 2)
                changed = True
        except Exception:
            # Keep pipeline robust: skip failed day backfill without interrupting report.
            continue

    return changed


def calc_last5_winrate_and_update_history(picks: List[Pick]) -> WinrateSummary:
    history_path = Path(__file__).with_name("nasdaq_reco_history.json")
    history = _load_history(history_path)
    if _backfill_est_winrate_5d(history):
        _save_history(history_path, history)

    today = dt.date.today().isoformat()
    symbols = [p.symbol for p in picks]
    high_est_threshold = 70.0

    # 先用历史中的前5个推荐日计算胜率（不含今天）
    prev_days = [r for r in history if r.get("date") != today]
    recent_days = prev_days[-5:]

    # 去重逻辑：在最近5个推荐日中，同一只股票只统计一次。
    # 统计口径使用该窗口内「最早被推荐那天」的 entry_price，与最新收盘价比较。
    earliest_entries: Dict[str, Dict[str, Any]] = {}
    # For "high-estimated-winrate" subset, keep earliest qualifying entry (>= threshold).
    earliest_high_est_entries: Dict[str, Dict[str, Any]] = {}
    for row in recent_days:
        day = str(row.get("date", ""))
        entries = row.get("picks", []) if isinstance(row.get("picks"), list) else []
        for e in entries:
            sym = str(e.get("symbol", "")).strip().upper()
            entry_price = float(e.get("entry_price", 0) or 0)
            est_winrate = float(e.get("est_winrate_5d", 0) or 0)
            if not sym or entry_price <= 0:
                continue
            # recent_days 按时间顺序（旧 -> 新），首次出现即为窗口内最早推荐
            if sym not in earliest_entries:
                earliest_entries[sym] = {"date": day, "entry_price": entry_price}
            if est_winrate >= high_est_threshold and sym not in earliest_high_est_entries:
                earliest_high_est_entries[sym] = {"date": day, "entry_price": entry_price, "est_winrate_5d": est_winrate}

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

    # 子集统计：最近5个推荐日中，曾出现“预估胜率>=70%”的股票（去重后每只仅统计一次）。
    high_est_symbols = sorted(earliest_high_est_entries.keys())
    high_est_total = 0
    high_est_wins = 0
    for sym in high_est_symbols:
        info = earliest_high_est_entries[sym]
        entry_price = float(info.get("entry_price", 0) or 0)
        now_price = current_close.get(sym)
        if entry_price <= 0 or now_price is None:
            continue
        high_est_total += 1
        if now_price > entry_price:
            high_est_wins += 1
    high_est_rate = round((high_est_wins / high_est_total * 100), 1) if high_est_total else 0.0

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
        "picks": [
            {
                "symbol": p.symbol,
                "entry_price": round(float(p.price), 4),
                "est_winrate_5d": round(float(p.signal_winrate_5d), 2),
                "confidence": int(p.confidence),
            }
            for p in picks
        ],
        "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    history = [r for r in history if r.get("date") != today]
    history.append(today_row)
    history = history[-60:]  # 保留最近60次
    _save_history(history_path, history)

    return WinrateSummary(
        total=total,
        wins=wins,
        rate=rate,
        details=details,
        high_est_total=high_est_total,
        high_est_wins=high_est_wins,
        high_est_rate=high_est_rate,
        high_est_threshold=high_est_threshold,
    )

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
            # 长线新闻：筛选后不再额外截断（有几条就输出几条）。
            news_items, news_stats = fetch_recent_news(symbol, name, limit=None)
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

