import os
import time
import random
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, date, time as dtime
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo

US_EASTERN = ZoneInfo("America/New_York")

def _make_session(total_retries: int = 3, backoff: float = 0.25) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.headers.update({"Connection": "keep-alive", "Accept": "application/json"})
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=50))
    return s

def _av_get(session: requests.Session, params: dict, api_key: str) -> dict | None:
    r = session.get("https://www.alphavantage.co/query", params={**params, "apikey": api_key}, timeout=15)
    if not r.ok:
        return None
    data = r.json()
    if isinstance(data, dict) and any(k in data for k in ("Note", "Error Message", "Information")):
        # brief jitter; caller may retry or back off
        time.sleep(0.6 + random.random() * 0.6)
        return None
    return data

def _fetch_intraday_1min_rth(session, sym, api_key, entitlement_mode: str):
    """Return (last_bar_ts_str, last_bar_close) from regular trading hours only."""
    data = _av_get(session, {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": sym,
        "interval": "1min",
        "outputsize": "compact",
        "extended_hours": "false",   # RTH only per requirement
        "entitlement": entitlement_mode,
    }, api_key)
    if not data:
        return None, None
    ts = data.get("Time Series (1min)", {})
    if not isinstance(ts, dict) or not ts:
        return None, None
    latest_ts = sorted(ts.keys())[-1]
    row = ts.get(latest_ts, {})
    close_val = row.get("4. close")
    return latest_ts, (float(close_val) if close_val not in (None, "") else None)

def _fetch_daily_close(session, sym, api_key, entitlement_mode: str):
    """Return (last_trading_date_str, daily_close_float)."""
    data = _av_get(session, {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": sym,
        "outputsize": "compact",     # enough to get latest row
        "entitlement": entitlement_mode,
    }, api_key)
    if not data:
        return None, None
    ts = data.get("Time Series (Daily)", {})
    if not isinstance(ts, dict) or not ts:
        return None, None
    last_date = sorted(ts.keys())[-1]
    row = ts[last_date]
    close_val = row.get("4. close") or row.get("5. adjusted close")
    return last_date, (float(close_val) if close_val not in (None, "") else None)

def _get_market_clock(now_et: datetime) -> dict:
    """Simple RTH gate. Treat 9:30–16:00 ET as regular session."""
    open_t = datetime.combine(now_et.date(), dtime(9, 30), tzinfo=US_EASTERN)
    close_t = datetime.combine(now_et.date(), dtime(16, 0), tzinfo=US_EASTERN)
    return {"now": now_et, "is_rth": open_t <= now_et <= close_t, "after_close": now_et > close_t}

def enrich_with_alpha_vantage(df_ready_input: pd.DataFrame, entitlement_mode: str = "delayed"):
    """
    Updated behavior:
      - Intraday uses extended_hours=false for RTH-only bars.
      - During RTH, price = latest 1-min RTH bar.
      - After 4:00 pm ET, first try TIME_SERIES_DAILY_ADJUSTED for today's official close.
        If that row is not yet published, fall back to the 16:00:00 ET intraday bar.
    """
    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not api_key:
        st.error("Alpha Vantage API key not found. Please set ALPHA_VANTAGE_KEY in your .env.")
        return None

    # caches (preserved)
    alpha_cache = st.session_state.setdefault("alpha_cache", {})

    df_ready = df_ready_input.copy()
    df_ready["__ticker_norm__"] = df_ready["ticker"].astype(str).str.strip().str.upper()
    unique_tickers = [t for t in sorted(set(df_ready["__ticker_norm__"])) if t]
    if not unique_tickers:
        st.warning("No valid tickers found to enrich.")
        return None

    # throttle guard for Premium 75
    window_start = time.monotonic()
    calls_in_window = 0
    def guard_call_window(limit=72):
        nonlocal window_start, calls_in_window
        now_m = time.monotonic()
        elapsed = now_m - window_start
        if elapsed >= 60:
            window_start = now_m
            calls_in_window = 0
        if calls_in_window >= limit:
            sleep_s = max(0.0, 60 - elapsed)
            st.info(f"API limit: pausing {sleep_s:.0f}s to stay under plan limit…")
            time.sleep(sleep_s)
            window_start = time.monotonic()
            calls_in_window = 0

    session = _make_session()
    now_et = datetime.now(tz=US_EASTERN)
    clock = _get_market_clock(now_et)

    progress = st.progress(0)
    status_text = st.empty()
    total = len(unique_tickers)

    results_price, results_date = {}, {}

    for i, sym in enumerate(unique_tickers, start=1):
        status_text.write(f"Fetching {sym} ({i}/{total}) …")

        px, px_date = None, None

        # After close: prefer official EOD from Daily Adjusted
        if clock["after_close"]:
            guard_call_window()
            d_date, d_close = _fetch_daily_close(session, sym, api_key, entitlement_mode)
            calls_in_window += 1
            if d_date == now_et.date().isoformat() and d_close is not None:
                px, px_date = d_close, d_date
            else:
                # Not posted yet, use 16:00 ET bar from RTH intraday
                guard_call_window()
                i_ts, i_close = _fetch_intraday_1min_rth(session, sym, api_key, entitlement_mode)
                calls_in_window += 1
                if i_ts and i_close is not None:
                    # Keep only the 16:00:00 bar if present; else latest RTH bar
                    try:
                        # Intraday timestamps are US/Eastern
                        last_dt = pd.to_datetime(i_ts).tz_localize(US_EASTERN, nonexistent="shift_forward", ambiguous="NaT")
                        if last_dt.time() != dtime(16, 0):
                            # try to find the exact 16:00 bar if available in the returned window
                            # fetch again but scan the time series via cache in the same call
                            data = _av_get(session, {
                                "function": "TIME_SERIES_INTRADAY",
                                "symbol": sym,
                                "interval": "1min",
                                "outputsize": "compact",
                                "extended_hours": "false",
                                "entitlement": entitlement_mode,
                            }, api_key)
                            if data and "Time Series (1min)" in data:
                                ts_map = data["Time Series (1min)"]
                                # find today's 16:00:00 ET
                                key_1600 = f"{now_et.date()} 16:00:00"
                                if key_1600 in ts_map:
                                    i_close = float(ts_map[key_1600]["4. close"])
                                    i_ts = key_1600
                                    calls_in_window += 0  # same request
                        px, px_date = i_close, i_ts
                    except Exception:
                        px, px_date = i_close, i_ts

        # During RTH: use latest 1-min RTH bar
        if px is None:
            guard_call_window()
            i_ts, i_close = _fetch_intraday_1min_rth(session, sym, api_key, entitlement_mode)
            calls_in_window += 1
            if i_close is not None:
                px, px_date = i_close, i_ts

        results_price[sym] = px
        results_date[sym] = px_date
        alpha_cache[sym] = {"market_price": px, "market_date": px_date} | alpha_cache.get(sym, {})

        progress.progress(int(i / total * 100))
        guard_call_window()

    progress.empty()
    status_text.empty()
    st.session_state["alpha_cache"] = alpha_cache

    # Build enriched base DataFrame
    df_enriched = df_ready.copy()
    df_enriched["market_price"] = df_enriched["__ticker_norm__"].map(results_price.get)
    df_enriched["market_date"] = df_enriched["__ticker_norm__"].map(results_date.get)

    df_enriched["quantity_num"] = pd.to_numeric(df_enriched["quantity"], errors="coerce")
    df_enriched["unit_cost_num"] = pd.to_numeric(df_enriched["unit_cost"], errors="coerce")
    df_enriched["market_price_num"] = pd.to_numeric(df_enriched["market_price"], errors="coerce")
    df_enriched["cost_basis"] = df_enriched["quantity_num"] * df_enriched["unit_cost_num"]
    df_enriched["market_value"] = df_enriched["quantity_num"] * df_enriched["market_price_num"]
    df_enriched["unrealized_gain_loss"] = df_enriched["market_value"] - df_enriched["cost_basis"]
    df_enriched["unrealized_gain_loss_pct"] = (df_enriched["unrealized_gain_loss"] / df_enriched["cost_basis"]) * 100

    st.session_state["df_enriched_base"] = df_enriched
    st.session_state["alpha_fetch_complete"] = True
    # Track last Alpha Vantage refresh time in ET (timezone-aware)
    try:
        st.session_state["alpha_last_refresh_et"] = datetime.now(tz=US_EASTERN)
    except Exception:
        pass
    return df_enriched


