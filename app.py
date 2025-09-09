import os
import time
import requests
import streamlit as st
from datetime import datetime, date, time
from datetime import timedelta as _timedelta
from datetime import datetime as _dt
from zoneinfo import ZoneInfo
from math import ceil as _ceil
import math
import streamlit.components.v1 as components
import pandas as pd
import datetime as dt
from functools import lru_cache
from dotenv import load_dotenv
from helpers import (
    _fmt_money,
    _fmt_money_signed,
    _fmt_qty,
    _fmt_pct_signed,
)
from alpha_vantage import enrich_with_alpha_vantage
from rule_catalog import load_rule_catalog, rule_meta, severity_badge_text
from ui_pages import page_upload_review_v2, page_overview, page_ai_assistant

# Using entitlement=delayed to pull 15-minute delayed US market data (premium 75 plan).
ENTITLEMENT_MODE = "delayed"

# Load environment variables from .env if present
load_dotenv()

# ===================== SHARED HELPERS: formatting + display builder =====================
import pandas as _pd
import numpy as _np

st.set_page_config(page_title="Trade Lifecycle Ops: STP & T+1 Settlement Readiness", page_icon="✅", layout="wide", initial_sidebar_state="expanded")

from datetime import date as _date_local

# ===================== Path helper and core routines (extracted) =====================
from pathlib import Path as _Path
import pandas as _pd
import numpy as _np
import os as _os
import random as _random
import requests as _requests

def _safe_ugl_pct(df):
    """Recompute unrealized_gain_loss_pct safely (no inf); in-place on df."""
    import numpy as _np
    import pandas as _pd
    if df is None or not isinstance(df, _pd.DataFrame) or df.empty:
        return df
    q  = _pd.to_numeric(df.get("quantity_num", df.get("Quantity")), errors="coerce").abs()
    uc = _pd.to_numeric(df.get("unit_cost_num", df.get("Unit Cost")), errors="coerce").abs()
    ugl = _pd.to_numeric(df.get("unrealized_gain_loss", df.get("Unrealized Gain/Loss")), errors="coerce")
    den = uc * q
    pct = _np.where((den > 0) & _np.isfinite(den) & _np.isfinite(ugl), (ugl.abs() / den) * 100.0, _np.nan)
    df["unrealized_gain_loss_pct"] = pct
    df.replace([_np.inf, -_np.inf], _np.nan, inplace=True)
    return df


def _ensure_enriched_base_from_latest():
    """
    Ensure st.session_state['df_enriched_base'] exists with numeric mirrors.
    Uses trades_clean.csv if available; else falls back to trades_raw.csv.
    """
    import pandas as _pd, numpy as _np
    from pathlib import Path as _Path
    data_dir = get_data_dir()
    df = st.session_state.get("df_enriched_base")
    if isinstance(df, _pd.DataFrame) and not df.empty:
        return df

    # Try cleaned trades first
    clean_p = _Path(data_dir) / "trades_clean.csv"
    raw_p   = _Path(data_dir) / "trades_raw.csv"
    if clean_p.exists():
        df = _pd.read_csv(clean_p)
    elif raw_p.exists():
        df = _pd.read_csv(raw_p)
    else:
        return None

    # Best-effort column resolution
    sid = None
    for c in ["security_id", "Security Id", "ticker", "symbol", "instrument"]:
        if c in df.columns:
            sid = c; break
    if sid is None:
        # fabricate a key if truly missing
        df["security_id"] = _pd.factorize(df.index)[0].astype(int).astype(str)
        sid = "security_id"

    # Resolve quantity as a Series and apply side sign
    qty_candidates = ["quantity_num", "Quantity", "qty"]
    _qty_series = None
    for _q in qty_candidates:
        if _q in df.columns:
            _qty_series = _pd.to_numeric(df[_q], errors="coerce")
            break
    if _qty_series is None:
        _qty_series = _pd.Series(_np.nan, index=df.index)
    side_series = None
    for _s in ["side", "Side"]:
        if _s in df.columns:
            side_series = df[_s].astype(str).str.upper().map({"BUY": 1, "SELL": -1})
            break
    if side_series is None:
        side_series = _pd.Series(1, index=df.index)
    qty = _pd.to_numeric(_qty_series, errors="coerce").fillna(0) * side_series.fillna(1)

    # Resolve execution/unit price as Series
    price_candidates = ["price", "Price", "unit_cost_num", "Unit Cost"]
    _price_series = None
    for _p in price_candidates:
        if _p in df.columns:
            _price_series = _pd.to_numeric(df[_p], errors="coerce")
            break
    if _price_series is None:
        _price_series = _pd.Series(0.0, index=df.index)
    price = _price_series.fillna(0)

    # Resolve last/market price; synthesize if missing
    last_candidates = ["last_price", "market_price_num", "Market Price"]
    _last_series = None
    for _l in last_candidates:
        if _l in df.columns:
            _last_series = _pd.to_numeric(df[_l], errors="coerce")
            break
    if _last_series is None or _last_series.isna().all():
        seed = int(st.session_state.get("last_demo_seed", 42))
        _np.random.seed(seed)
        shock = _np.random.normal(loc=1.0, scale=0.05, size=len(df))
        _last_series = price.abs() * _pd.Series(shock, index=df.index)
    last = _last_series.fillna(price.abs())

    base = _pd.DataFrame({
        "security_id": df[sid].astype(str),
        "quantity_num": _pd.to_numeric(qty, errors="coerce").fillna(0),
        "unit_cost_num": _pd.to_numeric(price, errors="coerce").fillna(0),
        "market_price_num": _pd.to_numeric(last, errors="coerce").fillna(0),
    })
    base["market_value"] = base["quantity_num"] * base["market_price_num"]
    base["unrealized_gain_loss"] = (base["market_price_num"] - base["unit_cost_num"]) * base["quantity_num"]
    _safe_ugl_pct(base)
    st.session_state["df_enriched_base"] = base
    # If a builder exists, use it for display cache
    builder = globals().get("build_enriched_display_frame")
    st.session_state["df_enriched_display"] = builder(base) if callable(builder) else base.copy()
    return base


def _apply_breaks_local(df, abs_thr, pct_thr):
    """Minimal break flagger if a global apply_breaks is not present."""
    import numpy as _np
    if df is None or df.empty:
        return df
    abs_thr = float(abs_thr)
    pct_thr = float(pct_thr)
    amt = _pd.to_numeric(df.get("unrealized_gain_loss"), errors="coerce").abs()
    pct = _pd.to_numeric(df.get("unrealized_gain_loss_pct"), errors="coerce").abs()
    cond = (_np.isfinite(amt) & (amt >= abs_thr)) | (_np.isfinite(pct) & (pct >= pct_thr))
    df["break_flag"] = _np.where(cond, "🚨", "✅")
    df["break_reason"] = _np.where((amt >= abs_thr) & _np.isfinite(amt), "Unrealized $ ≥ threshold",
                          _np.where((pct >= pct_thr) & _np.isfinite(pct), "Unrealized % ≥ threshold", "Within threshold"))
    df["break_reason_category"] = _np.where(df["break_flag"] == "🚨", "Threshold", "OK")
    return df

def get_data_dir() -> _Path:
    """Resolve the data directory, overridable via st.session_state["DATA_DIR"]."""
    try:
        override = st.session_state.get("DATA_DIR")
        if override:
            p = _Path(override)
            p.mkdir(parents=True, exist_ok=True)
            return p
    except Exception:
        pass
    p = _Path(__file__).resolve().parent / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p

# ===================== US Holiday Calendar (cached loaders + helpers) =====================
@st.cache_data(show_spinner=False)
def load_us_holiday_df(csv_path: str = "data/us_equities_holidays_2025.csv") -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df = df.dropna(subset=["Date"]).reset_index(drop=True)
        return df
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_us_holiday_set(csv_path: str = "data/us_equities_holidays_2025.csv") -> set[str]:
    df = load_us_holiday_df(csv_path)
    if df is None or df.empty:
        return set()
    return {d.isoformat() for d in df["Date"].tolist()}

def _to_date(value) -> dt.date | None:
    if isinstance(value, dt.date):
        return value
    try:
        return pd.to_datetime(value, errors="coerce").date()
    except Exception:
        return None

def weekend_or_holiday_flag(settlement_date_value, holiday_iso_set: set[str]) -> tuple[bool, str | None]:
    d = _to_date(settlement_date_value)
    if d is None:
        return False, None
    if d.weekday() >= 5:
        return True, "Settlement date falls on a weekend/holiday"
    if d.isoformat() in holiday_iso_set:
        return True, "Settlement date falls on a weekend/holiday"
    return False, None

def apply_weekend_or_holiday_rule(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return trades_df
    if "settlement_date" not in trades_df.columns:
        return trades_df
    flags: list[bool] = []
    reasons: list[str | None] = []
    for val in trades_df["settlement_date"].tolist():
        bad, reason = weekend_or_holiday_flag(val, holiday_set)
        flags.append(bad)
        reasons.append(reason if bad else None)
    out = trades_df.copy()
    out["weekend_or_holiday_settlement"] = flags
    if "issue" in out.columns:
        out["issue"] = out["issue"].astype("object")
        out["issue"] = out.apply(
            lambda r: (str(r["issue"]) + " | " + reasons[r.name]) if (r["issue"] not in [None, "", "nan"] and flags[r.name]) else
                      (reasons[r.name] if flags[r.name] and not r["issue"] else r["issue"]),
            axis=1
        )
    else:
        out["issue"] = [reasons[i] if flags[i] else None for i in range(len(flags))]
    return out

# Initialize holiday assets at import-time so pages can reference them
holiday_df = load_us_holiday_df()
holiday_set = load_us_holiday_set()

def render_holiday_badge():
    if holiday_df is None or (hasattr(holiday_df, "empty") and holiday_df.empty):
        st.info("Holiday file loaded: 0 dates")
    else:
        st.info(f"Holiday file loaded: {len(holiday_df)} dates")

def generate_security_master(recreate: bool = False) -> _Path | None:
    data_dir = get_data_dir()
    csv_path = data_dir / "security_master.csv"
    if csv_path.exists() and not recreate:
        return csv_path

    # Seed tickers and MIC mapping
    xnas_syms = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
    xnys_syms = ["JPM", "BAC", "XOM", "UNH", "T"]

    default_names = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com, Inc.",
        "GOOGL": "Alphabet Inc.",
        "META": "Meta Platforms, Inc.",
        "NVDA": "NVIDIA Corporation",
        "TSLA": "Tesla, Inc.",
        "JPM": "JPMorgan Chase & Co.",
        "BAC": "Bank of America Corporation",
        "XOM": "Exxon Mobil Corporation",
        "UNH": "UnitedHealth Group Incorporated",
        "T": "AT&T Inc.",
    }

    def _mic_for(sym: str) -> str:
        return "XNAS" if sym in xnas_syms else ("XNYS" if sym in xnys_syms else "")

    base_rows: list[dict] = []
    for sym in xnas_syms + xnys_syms:
        base_rows.append(
            {
                "security_id": sym,
                "company_name": default_names.get(sym, ""),
                "asset_class": "Equity",
                "currency": "USD",
                "country": "US",
                "mic": _mic_for(sym),
                "settlement_cycle": "T+1",
                "sector": "",
                "active_flag": True,
            }
        )

    # Optional enrichment via Alpha Vantage OVERVIEW
    def _fetch_overview(symbol: str, api_key: str) -> dict | None:
        try:
            resp = _requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "OVERVIEW",
                    "symbol": symbol,
                    "apikey": api_key,
                },
                timeout=20,
            )
            if not resp.ok:
                return None
            data = resp.json()
            return data if isinstance(data, dict) and data else None
        except Exception:
            return None

    api_key = _os.getenv("ALPHA_VANTAGE_KEY")
    if api_key:
        for row in base_rows:
            info = _fetch_overview(row["security_id"], api_key)
            # Small courtesy delay avoided here; upstream fetchers respect rate limits elsewhere
            if isinstance(info, dict) and info:
                name_val = info.get("Name")
                sector_val = info.get("Sector")
                if isinstance(name_val, str) and name_val.strip():
                    row["company_name"] = name_val.strip()
                if isinstance(sector_val, str) and sector_val.strip():
                    row["sector"] = sector_val.strip()

    cols = [
        "security_id",
        "company_name",
        "asset_class",
        "currency",
        "country",
        "mic",
        "settlement_cycle",
        "sector",
        "active_flag",
    ]
    df_sec = _pd.DataFrame(base_rows, columns=cols)
    df_sec.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path

def generate_ssi_master(recreate: bool = False) -> _Path | None:
    data_dir = get_data_dir()
    ssi_csv_path = data_dir / "counterparty_ssi.csv"
    if ssi_csv_path.exists() and not recreate:
        return ssi_csv_path

    today_str = _date_local.today().isoformat()
    rows_ssi = [
        {"counterparty_legal_name": "Goldman Sachs & Co. LLC", "lei": "LEI-GSCO-0000000001", "market": "US Equities", "depository": "DTC", "custodian_bic": "GSDMUS33XXX", "depository_account": "DTC-0051", "cash_account": "ABA-026009593", "currency": "USD", "location_city": "New York", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "Goldman Sachs & Co. LLC", "lei": "LEI-GSCO-0000000001", "market": "US Equities", "depository": "DTC", "custodian_bic": "GSDMUS33XXX", "depository_account": "DTC-0151", "cash_account": "ABA-021000021", "currency": "USD", "location_city": "Chicago", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "Morgan Stanley & Co. LLC", "lei": "LEI-MSCI-0000000002", "market": "US Equities", "depository": "DTC", "custodian_bic": "MSDMUS33XXX", "depository_account": "DTC-0052", "cash_account": "ABA-111000025", "currency": "USD", "location_city": "New York", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "Morgan Stanley & Co. LLC", "lei": "LEI-MSCI-0000000002", "market": "US Equities", "depository": "DTC", "custodian_bic": "MSDMUS33XXX", "depository_account": "DTC-0152", "cash_account": "ABA-031100209", "currency": "USD", "location_city": "San Francisco", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "J.P. Morgan Securities LLC", "lei": "LEI-JPMS-0000000003", "market": "US Equities", "depository": "DTC", "custodian_bic": "JPMXUS33XXX", "depository_account": "DTC-0061", "cash_account": "ABA-053000219", "currency": "USD", "location_city": "New York", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "BofA Securities, Inc.", "lei": "LEI-BOFA-0000000004", "market": "US Equities", "depository": "DTC", "custodian_bic": "BOFAXUS33XXX", "depository_account": "DTC-0065", "cash_account": "ABA-122105155", "currency": "USD", "location_city": "Charlotte", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "Barclays Capital Inc.", "lei": "LEI-BARC-0000000005", "market": "US Equities", "depository": "DTC", "custodian_bic": "BARXUS33XXX", "depository_account": "DTC-0070", "cash_account": "IBAN-FAKE123456789", "currency": "USD", "location_city": "New York", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "Citigroup Global Markets Inc.", "lei": "LEI-CITI-0000000006", "market": "US Equities", "depository": "DTC", "custodian_bic": "CITXUS33XXX", "depository_account": "DTC-0072", "cash_account": "IBAN-FAKE987654321", "currency": "USD", "location_city": "New York", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "UBS Securities LLC", "lei": "LEI-UBSS-0000000007", "market": "US Equities", "depository": "DTC", "custodian_bic": "UBSXUS33XXX", "depository_account": "DTC-0081", "cash_account": "ABA-026009593", "currency": "USD", "location_city": "New York", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "Deutsche Bank Securities Inc.", "lei": "LEI-DBSI-0000000008", "market": "US Equities", "depository": "DTC", "custodian_bic": "DBSXUS33XXX", "depository_account": "DTC-0085", "cash_account": "ABA-021000021", "currency": "USD", "location_city": "New York", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "Jefferies LLC", "lei": "LEI-JEFF-0000000009", "market": "US Equities", "depository": "DTC", "custodian_bic": "JEFXUS33XXX", "depository_account": "DTC-0090", "cash_account": "ABA-026013673", "currency": "USD", "location_city": "Boston", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "BNP Paribas Securities Corp.", "lei": "LEI-BNPP-0000000010", "market": "US Equities", "depository": "DTC", "custodian_bic": "BNPXUS33XXX", "depository_account": "DTC-0093", "cash_account": "ABA-031100209", "currency": "USD", "location_city": "New York", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "Wells Fargo Securities, LLC", "lei": "LEI-WFSC-0000000011", "market": "US Equities", "depository": "DTC", "custodian_bic": "WFSXUS33XXX", "depository_account": "DTC-0095", "cash_account": "ABA-122105155", "currency": "USD", "location_city": "Charlotte", "location_country": "US", "effective_date": today_str, "active_flag": True},
        {"counterparty_legal_name": "Interactive Brokers LLC", "lei": "LEI-IBKR-0000000012", "market": "US Equities", "depository": "DTC", "custodian_bic": "IBKRUS33XXX", "depository_account": "DTC-0101", "cash_account": "IBAN-FAKE246813579", "currency": "USD", "location_city": "Chicago", "location_country": "US", "effective_date": today_str, "active_flag": True},
    ]

    cols_ssi = [
        "counterparty_legal_name",
        "lei",
        "market",
        "depository",
        "custodian_bic",
        "depository_account",
        "cash_account",
        "currency",
        "location_city",
        "location_country",
        "effective_date",
        "active_flag",
    ]
    _pd.DataFrame(rows_ssi, columns=cols_ssi).to_csv(ssi_csv_path, index=False, encoding="utf-8")
    return ssi_csv_path

def generate_trades_raw(recreate: bool = False, seed: int | None = None) -> _Path | None:
    data_dir = get_data_dir()
    trades_csv_path = data_dir / "trades_raw.csv"
    if trades_csv_path.exists() and not recreate:
        return trades_csv_path

    sec_path = data_dir / "security_master.csv"
    ssi_path = data_dir / "counterparty_ssi.csv"
    if not sec_path.exists() or not ssi_path.exists():
        raise FileNotFoundError("Missing masters. Run Security Master and Counterparty SSI Master first.")
    if not _os.getenv("ALPHA_VANTAGE_KEY"):
        raise RuntimeError("Alpha Vantage API key not found. Set ALPHA_VANTAGE_KEY and restart.")

    sec = _pd.read_csv(sec_path)
    ssi = _pd.read_csv(ssi_path)
    if "active_flag" in sec.columns:
        sec = sec[sec["active_flag"] == True]
    if "asset_class" in sec.columns:
        sec = sec[sec["asset_class"].astype(str) == "Equity"]
    if "active_flag" in ssi.columns:
        ssi = ssi[ssi["active_flag"] == True]
    if sec.empty or ssi.empty:
        raise RuntimeError("Masters contain no active rows.")

    # Prepare minimal frame for price prefetch via existing wrapper
    tickers_series = sec["security_id"].astype(str).str.strip().str.upper()
    unique_tickers = sorted(list(dict.fromkeys([t for t in tickers_series.tolist() if t])))
    if not unique_tickers:
        raise RuntimeError("No equity tickers available for pricing.")
    df_ready = _pd.DataFrame({"ticker": unique_tickers, "quantity": [1] * len(unique_tickers), "unit_cost": [0] * len(unique_tickers)})

    df_px = enrich_with_alpha_vantage(df_ready, ENTITLEMENT_MODE)
    if not isinstance(df_px, _pd.DataFrame) or df_px.empty:
        raise RuntimeError("Failed to fetch prices from Alpha Vantage.")

    px_by_ticker, date_by_ticker = {}, {}
    for _, r in df_px.iterrows():
        sym = str(r.get("__ticker_norm__", "")).upper().strip()
        px = r.get("market_price_num", r.get("market_price"))
        dt_raw = r.get("market_date")
        if not sym:
            continue
        try:
            px_val = float(px) if px not in (None, ""
            ) else None
        except Exception:
            px_val = None
        if px_val is None or px_val <= 0:
            continue
        dt_str = ""
        if isinstance(dt_raw, str) and dt_raw:
            dt_str = dt_raw[:10]
        else:
            try:
                dt_str = _pd.to_datetime(dt_raw).strftime("%Y-%m-%d")
            except Exception:
                dt_str = ""
        if not dt_str:
            continue
        px_by_ticker[sym] = px_val
        date_by_ticker[sym] = dt_str

    priced_syms = list(px_by_ticker.keys())
    if len(priced_syms) < 6:
        raise RuntimeError("Insufficient priced tickers (need at least 6). Try again later.")

    # Settlement cycle map
    sc_map = {}
    if {"security_id", "settlement_cycle"}.issubset(sec.columns):
        for _, r in sec.iterrows():
            sc_map[str(r["security_id"]).upper()] = str(r["settlement_cycle"]).upper()

    cpys = ssi["counterparty_legal_name"].astype(str).tolist()
    qty_choices = [100, 200, 500, 1000, 2500, 5000]

    if seed is not None:
        _random.seed(int(seed))
    else:
        _random.seed(42)

    # Attempt ADV-based weighting from any cached daily series (optional)
    try:
        daily_cache = st.session_state.get("daily_cache", {})
    except Exception:
        daily_cache = {}
    def _adv_for(sym: str) -> float:
        ts = daily_cache.get(sym)
        if isinstance(ts, dict) and ts:
            try:
                keys = sorted(ts.keys())
                tail_keys = keys[-20:]
                vols = []
                for k in tail_keys:
                    row = ts.get(k, {})
                    v = row.get("6. volume") or row.get("volume")
                    if v not in (None, ""):
                        vols.append(float(v))
                if vols:
                    return max(1.0, float(sum(vols) / len(vols)))
            except Exception:
                return 1.0
        return 1.0
    adv_by_ticker = {t: _adv_for(t) for t in priced_syms}

    # Business day helper
    from datetime import date as _date, timedelta as _td
    def add_business_days(d: _date, n: int) -> _date:
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        cur = d
        while remaining > 0:
            cur = cur + _td(days=step)
            if cur.weekday() < 5:
                remaining -= 1
        return cur

    TARGET_VALID = 16
    TICKER_CAP_RATIO = 0.25
    CPY_CAP = 4
    MIN_TICKERS = min(8, len(priced_syms))
    max_per_ticker = max(2, int(TARGET_VALID * TICKER_CAP_RATIO))
    ranked = sorted(priced_syms, key=lambda t: adv_by_ticker.get(t, 1.0), reverse=True)
    planned_tickers: list[str] = []
    ticker_counts: dict[str, int] = {t: 0 for t in priced_syms}
    for t in ranked[:MIN_TICKERS]:
        planned_tickers.append(t)
        ticker_counts[t] += 1
    while len(planned_tickers) < TARGET_VALID:
        candidates = [t for t in priced_syms if ticker_counts.get(t, 0) < max_per_ticker] or priced_syms[:]
        weights = [adv_by_ticker.get(t, 1.0) for t in candidates]
        try:
            choice = _random.choices(candidates, weights=weights, k=1)[0]
        except Exception:
            choice = _random.choice(candidates)
        planned_tickers.append(choice)
        ticker_counts[choice] = ticker_counts.get(choice, 0) + 1

    counterparties = list(dict.fromkeys(cpys)) or ["Nonexistent Broker LLC"]
    counterparty_counts: dict[str, int] = {c: 0 for c in counterparties}
    def _assign_counterparty() -> str:
        under_cap = [c for c in counterparties if counterparty_counts.get(c, 0) < CPY_CAP]
        pool = under_cap if under_cap else counterparties
        c = min(pool, key=lambda x: counterparty_counts.get(x, 0))
        counterparty_counts[c] = counterparty_counts.get(c, 0) + 1
        return c

    rows_valid = []
    for sym in planned_tickers:
        px = px_by_ticker.get(sym)
        td_str = date_by_ticker.get(sym)
        if px is None or not td_str:
            continue
        td = _pd.to_datetime(td_str).date()
        sc = sc_map.get(sym, "T+1")
        bd_add = 2 if sc == "T+2" else 1
        sd = add_business_days(td, bd_add)
        rows_valid.append({
            "trade_date": td.strftime("%Y-%m-%d"),
            "settlement_date": sd.strftime("%Y-%m-%d"),
            "side": _random.choice(["Buy", "Sell"]),
            "quantity": int(_random.choice(qty_choices)),
            "security_id": sym,
            "price": float(px),
            "counterparty_legal_name": _assign_counterparty(),
            "dq_seed_issue": "",
            "is_seed_bad": False,
        })

    rows_bad = []
    issues = [
        "missing_security_id",
        "quantity_zero",
        "missing_price",
        "invalid_side",
        "unknown_counterparty",
        "bad_settlement_cycle",
        "weekend_settlement",
        "missing_counterparty",
    ]
    ranked_syms = ranked if ranked else (priced_syms or ["AAPL"])  # fallback
    for idx, code in enumerate(issues):
        sym = ranked_syms[idx % len(ranked_syms)]
        td_str = date_by_ticker.get(sym, _date.today().strftime("%Y-%m-%d"))
        td = _pd.to_datetime(td_str).date()
        cpty = counterparties[idx % len(counterparties)]
        px_val = px_by_ticker.get(sym, _pd.NA)
        if code == "missing_security_id":
            rows_bad.append({"trade_date": td.strftime("%Y-%m-%d"), "settlement_date": add_business_days(td, 1).strftime("%Y-%m-%d"), "side": "Buy", "quantity": 100, "security_id": "", "price": _pd.NA, "counterparty_legal_name": cpty, "dq_seed_issue": code, "is_seed_bad": True})
        elif code == "quantity_zero":
            rows_bad.append({"trade_date": td.strftime("%Y-%m-%d"), "settlement_date": add_business_days(td, 1).strftime("%Y-%m-%d"), "side": "Sell", "quantity": 0, "security_id": sym, "price": float(px_val) if px_val is not _pd.NA else _pd.NA, "counterparty_legal_name": cpty, "dq_seed_issue": code, "is_seed_bad": True})
        elif code == "missing_price":
            rows_bad.append({"trade_date": td.strftime("%Y-%m-%d"), "settlement_date": add_business_days(td, 1).strftime("%Y-%m-%d"), "side": "Buy", "quantity": 500, "security_id": sym, "price": _pd.NA, "counterparty_legal_name": cpty, "dq_seed_issue": code, "is_seed_bad": True})
        elif code == "invalid_side":
            rows_bad.append({"trade_date": td.strftime("%Y-%m-%d"), "settlement_date": add_business_days(td, 1).strftime("%Y-%m-%d"), "side": "BUY", "quantity": 1000, "security_id": sym, "price": float(px_val) if px_val is not _pd.NA else _pd.NA, "counterparty_legal_name": cpty, "dq_seed_issue": code, "is_seed_bad": True})
        elif code == "unknown_counterparty":
            rows_bad.append({"trade_date": td.strftime("%Y-%m-%d"), "settlement_date": add_business_days(td, 1).strftime("%Y-%m-%d"), "side": "Sell", "quantity": 200, "security_id": sym, "price": float(px_val) if px_val is not _pd.NA else _pd.NA, "counterparty_legal_name": "Nonexistent Broker LLC", "dq_seed_issue": code, "is_seed_bad": True})
        elif code == "bad_settlement_cycle":
            rows_bad.append({"trade_date": td.strftime("%Y-%m-%d"), "settlement_date": add_business_days(td, 2).strftime("%Y-%m-%d"), "side": "Buy", "quantity": 2500, "security_id": sym, "price": float(px_val) if px_val is not _pd.NA else _pd.NA, "counterparty_legal_name": cpty, "dq_seed_issue": code, "is_seed_bad": True})
        elif code == "weekend_settlement":
            sat = td
            while sat.weekday() != 5:
                sat = sat + _td(days=1)
            rows_bad.append({"trade_date": td.strftime("%Y-%m-%d"), "settlement_date": sat.strftime("%Y-%m-%d"), "side": "Sell", "quantity": 5000, "security_id": sym, "price": float(px_val) if px_val is not _pd.NA else _pd.NA, "counterparty_legal_name": cpty, "dq_seed_issue": code, "is_seed_bad": True})
        elif code == "missing_counterparty":
            rows_bad.append({"trade_date": td.strftime("%Y-%m-%d"), "settlement_date": add_business_days(td, 1).strftime("%Y-%m-%d"), "side": "Buy", "quantity": 100, "security_id": sym, "price": float(px_val) if px_val is not _pd.NA else _pd.NA, "counterparty_legal_name": "", "dq_seed_issue": code, "is_seed_bad": True})

    all_rows = rows_valid + rows_bad[:8]
    if len(all_rows) < 24:
        needed = 24 - len(all_rows)
        all_rows.extend(rows_valid[:needed])
    _random.shuffle(all_rows)

    cols_trades = [
        "trade_date",
        "settlement_date",
        "side",
        "quantity",
        "security_id",
        "price",
        "counterparty_legal_name",
        "dq_seed_issue",
        "is_seed_bad",
    ]
    df_trades = _pd.DataFrame(all_rows, columns=cols_trades).head(24)
    df_trades.to_csv(trades_csv_path, index=False, encoding="utf-8")
    return trades_csv_path

def run_validation_kpis() -> dict:
    data_dir = get_data_dir()
    sec_path = data_dir / "security_master.csv"
    ssi_path = data_dir / "counterparty_ssi.csv"
    trd_path = data_dir / "trades_raw.csv"
    if not (sec_path.exists() and ssi_path.exists() and trd_path.exists()):
        missing = [p.name for p in [sec_path, ssi_path, trd_path] if not p.exists()]
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")

    sec = _pd.read_csv(sec_path)
    ssi = _pd.read_csv(ssi_path)
    trd = _pd.read_csv(trd_path)
    if "active_flag" in sec.columns:
        sec = sec[sec["active_flag"] == True]
    if "active_flag" in ssi.columns:
        ssi = ssi[ssi["active_flag"] == True]

    def _norm_str_series(s: _pd.Series) -> _pd.Series:
        return s.astype(str).str.strip()

    valid_secs = set(_norm_str_series(sec.get("security_id", _pd.Series(dtype=str))))
    settle_cycle_by_sec = {}
    if "security_id" in sec.columns and "settlement_cycle" in sec.columns:
        settle_cycle_by_sec = {
            str(r["security_id"]).strip(): str(r["settlement_cycle"]).strip()
            for _, r in sec[["security_id", "settlement_cycle"]].dropna(subset=["security_id"]).iterrows()
        }

    valid_cpys = set(_norm_str_series(ssi.get("counterparty_legal_name", _pd.Series(dtype=str))))

    ssi_us = ssi.copy()
    if "market" in ssi_us.columns:
        ssi_us = ssi_us[ssi_us["market"].astype(str).str.strip() == "US Equities"]
    req_cols = ["counterparty_legal_name", "depository_account", "cash_account"]
    for c in req_cols:
        if c not in ssi_us.columns:
            ssi_us[c] = _pd.NA
    ssi_us["__has_req"] = ssi_us["depository_account"].notna() & ssi_us["cash_account"].notna()
    ssi_complete_by_cpy = (
        ssi_us.groupby("counterparty_legal_name")["__has_req"].max().to_dict()
        if "counterparty_legal_name" in ssi_us.columns
        else {}
    )

    from datetime import date as _date, timedelta as _td
    def add_business_days(d: _date, n: int) -> _date:
        step = 1 if n >= 0 else -1
        days_to_add = abs(n)
        cur = d
        while days_to_add > 0:
            cur = cur + _td(days=step)
            if cur.weekday() < 5:
                days_to_add -= 1
        return cur

    df = trd.copy()
    for col in ["security_id", "side", "counterparty_legal_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in ["trade_date", "settlement_date"]:
        if col in df.columns:
            df[col] = _pd.to_datetime(df[col], errors="coerce").dt.date

    required_fields = ["trade_date", "settlement_date", "side", "quantity", "security_id", "price", "counterparty_legal_name"]

    def _row_rule_check(row):
        for rf in required_fields:
            if rf not in row or _pd.isna(row[rf]) or (isinstance(row[rf], str) and not str(row[rf]).strip()):
                return ("presence_missing_field", "Missing required field", False, False)
        try:
            qty = float(row["quantity"]) if row["quantity"] is not None else None
        except Exception:
            qty = None
        if qty is None or qty <= 0:
            return ("quantity_nonpositive", "Quantity must be > 0", False, False)
        try:
            price = float(row["price"]) if row["price"] is not None else None
        except Exception:
            price = None
        if price is None or price <= 0:
            return ("price_nonpositive", "Price must be > 0", False, False)
        side = str(row["side"]).strip()
        if side not in {"Buy", "Sell"}:
            return ("invalid_side", "Side must be Buy or Sell", False, False)

        sec_id = str(row["security_id"]).strip()
        if sec_id not in valid_secs:
            return ("unknown_security", "Security not found or inactive", True, False)
        cpy = str(row["counterparty_legal_name"]).strip()
        if cpy not in valid_cpys:
            return ("unknown_counterparty", "Counterparty not found or inactive", True, False)
        if not bool(ssi_complete_by_cpy.get(cpy, False)):
            return ("incomplete_ssi", "Missing SSI depository/cash details for US Equities", True, False)
        cycle = settle_cycle_by_sec.get(sec_id, "").upper().replace(" ", "")
        td = row["trade_date"]
        sd = row["settlement_date"]
        if isinstance(td, _date) and isinstance(sd, _date):
            if cycle in {"T+1", "T+2"}:
                n = 1 if cycle == "T+1" else 2
                expected = add_business_days(td, n)
                if sd != expected:
                    return ("bad_settlement_cycle", "Settlement date does not match settlement cycle", True, False)
        # Weekend/Holiday calendar rule (US)
        bad_flag, reason_txt = weekend_or_holiday_flag(sd, holiday_set)
        if bad_flag:
            return ("weekend_or_holiday_settlement", "Settlement date falls on a weekend/holiday", True, False)
        return (None, None, True, True)

    results = df.apply(_row_rule_check, axis=1, result_type="expand")
    results.columns = ["Rule Code", "Exception Reason", "_pass_r1_r2", "stp_ready"]
    df = _pd.concat([df, results], axis=1)
    # Attach YAML-backed metadata without changing detection logic
    try:
        _cat = []
        _sev = []
        _own = []
        _reason = []
        _sev_label = []
        for _code, _reason_raw in zip(df.get("Rule Code", _pd.Series([None] * len(df))), df.get("Exception Reason", _pd.Series([None] * len(df)))):
            meta = rule_meta(_code)
            cat = meta.get("category") if isinstance(meta, dict) else None
            sev = meta.get("severity") if isinstance(meta, dict) else None
            own = meta.get("owner") if isinstance(meta, dict) else None
            msg = meta.get("message") if isinstance(meta, dict) else None
            _cat.append(cat)
            _sev.append(sev)
            _own.append(own)
            _reason.append(msg if (isinstance(msg, str) and len(msg.strip()) > 0) else _reason_raw)
            _sev_label.append(severity_badge_text(sev))
        if len(_cat) == len(df):
            df["Category"] = _cat
            df["Severity"] = _sev
            df["Owner"] = _own
            df["Exception Reason"] = _reason
            df["Severity Label"] = _sev_label
    except Exception:
        # Fail gracefully; keep original columns
        pass

    def _mk(row: _pd.Series) -> str:
        td = row.get("trade_date")
        from datetime import date as __date
        td_str = td.isoformat() if isinstance(td, __date) else str(td)
        side = str(row.get("side", "")).strip()
        sec_id = str(row.get("security_id", "")).strip()
        qty = row.get("quantity")
        try:
            qty_str = str(int(qty)) if float(qty).is_integer() else str(qty)
        except Exception:
            qty_str = str(qty)
        try:
            pr = float(row.get("price", _pd.NA))
            price_str = f"{pr:.2f}"
        except Exception:
            price_str = str(row.get("price", ""))
        cpy = str(row.get("counterparty_legal_name", "")).strip()
        return f"{td_str}|{side}|{sec_id}|{qty_str}|{price_str}|{cpy}"
    df["match_key"] = df.apply(_mk, axis=1)

    df_clean = df[df["stp_ready"] == True].copy()
    df_exceptions = df[df["stp_ready"] == False].copy()

    total = len(df)
    schema_pass = int(df["_pass_r1_r2"].sum()) if "_pass_r1_r2" in df.columns else 0
    stp_ready_count = int(df["stp_ready"].sum()) if "stp_ready" in df.columns else 0
    pct_schema = (schema_pass / total * 100.0) if total else 0.0
    pct_stp = (stp_ready_count / total * 100.0) if total else 0.0
    exc_by_rule = (
        df_exceptions.groupby(["Rule Code", "Exception Reason"]).size().reset_index(name="Count")
        if len(df_exceptions) > 0
        else _pd.DataFrame({"Rule Code": [], "Exception Reason": [], "Count": []})
    )

    # Persist outputs
    df_clean.to_csv(data_dir / "trades_clean.csv", index=False, encoding="utf-8")
    df_exceptions.to_csv(data_dir / "trades_exceptions.csv", index=False, encoding="utf-8")

    # Drop helper column before returning metrics
    if "_pass_r1_r2" in df.columns:
        df.drop(columns=["_pass_r1_r2"], inplace=True)

    return {
        "records_ingested": int(total),
        "pct_passing_schema": float(round(pct_schema, 2)),
        "pct_stp_ready": float(round(pct_stp, 2)),
        "exceptions_by_rule": exc_by_rule,
    }

def generate_broker_confirms(recreate: bool = False) -> _Path | None:
    data_dir = get_data_dir()
    trd_path = data_dir / "trades_raw.csv"
    ssi_path = data_dir / "counterparty_ssi.csv"
    out_path = data_dir / "broker_confirms.csv"
    if out_path.exists() and not recreate:
        return out_path
    if not trd_path.exists() or not ssi_path.exists():
        raise FileNotFoundError("Missing masters/trades. Run Steps 1B and 1C first.")
    trd = _pd.read_csv(trd_path)
    ssi = _pd.read_csv(ssi_path)
    required_fields = ["trade_date", "settlement_date", "side", "quantity", "security_id", "price", "counterparty_legal_name"]
    for c in required_fields:
        if c not in trd.columns:
            raise RuntimeError("Trades file missing required columns. Regenerate Step 1C.")
    if "active_flag" in ssi.columns:
        ssi = ssi[ssi["active_flag"] == True]
    valid_cpys = set(ssi.get("counterparty_legal_name", _pd.Series(dtype=str)).astype(str).str.strip())

    pool = trd.copy()
    for c in required_fields:
        pool = pool[pool[c].notna()]
    pool = pool[pool["side"].astype(str).str.strip().isin(["Buy", "Sell"])]
    pool = pool[pool["counterparty_legal_name"].astype(str).str.strip().isin(valid_cpys)]
    if pool.empty:
        raise RuntimeError("No eligible trades found to generate mock broker confirms.")

    # MIC map
    sec_path = data_dir / "security_master.csv"
    mic_map = {}
    if sec_path.exists():
        try:
            _sec = _pd.read_csv(sec_path)
            if {"security_id", "mic"}.issubset(_sec.columns):
                mic_map = {str(r["security_id"]).strip(): str(r["mic"]).strip() for _, r in _sec[["security_id", "mic"]].dropna(subset=["security_id"]).iterrows()}
        except Exception:
            mic_map = {}

    _random.seed(42)
    n_min, n_max = 16, 24
    n_pool = len(pool)
    n_pick = min(max(n_min, min(n_pool, _random.randint(n_min, n_max))), n_pool)

    pool["security_id"] = pool["security_id"].astype(str).str.strip()
    pool["counterparty_legal_name"] = pool["counterparty_legal_name"].astype(str).str.strip()
    tickers = list(dict.fromkeys(pool["security_id"].tolist()))
    counterparties = list(dict.fromkeys(pool["counterparty_legal_name"].tolist()))

    chosen_idx = []
    # cover tickers
    for t in tickers:
        if len(chosen_idx) >= n_pick:
            break
        sub = pool[pool["security_id"] == t]
        if not sub.empty:
            chosen_idx.append(sub.sample(1, random_state=_random.randint(0, 10_000)).index[0])
    # cover counterparties
    for c in counterparties:
        if len(chosen_idx) >= n_pick:
            break
        sub = pool[pool["counterparty_legal_name"] == c]
        sub = sub[~sub.index.isin(chosen_idx)]
        if not sub.empty:
            chosen_idx.append(sub.sample(1, random_state=_random.randint(0, 10_000)).index[0])
    # fill remaining
    remaining = pool.loc[~pool.index.isin(chosen_idx)]
    if not remaining.empty and len(chosen_idx) < n_pick:
        extra = remaining.sample(n_pick - len(chosen_idx), random_state=123, replace=False)
        chosen_idx.extend(extra.index.tolist())

    sample = pool.loc[chosen_idx].copy()
    def _gen_id(i: int) -> str:
        return f"BCF-{i:06d}"
    confirms_rows = []
    for i, (_, r) in enumerate(sample.iterrows(), start=1):
        sec_id = str(r.get("security_id", "")).strip()
        confirms_rows.append({
            "confirm_id": _gen_id(i),
            "trade_date": str(r.get("trade_date", ""))[:10],
            "side": str(r.get("side", "")),
            "quantity": int(float(r.get("quantity", 0)) if str(r.get("quantity", "")).strip() != "" else 0),
            "security_id": sec_id,
            "price": float(r.get("price", 0.0) or 0.0),
            "counterparty_legal_name": str(r.get("counterparty_legal_name", "")),
            "venue_mic": mic_map.get(sec_id, ""),
        })
    cols = ["confirm_id", "trade_date", "side", "quantity", "security_id", "price", "counterparty_legal_name", "venue_mic"]
    _pd.DataFrame(confirms_rows, columns=cols).to_csv(out_path, index=False, encoding="utf-8")
    return out_path

def run_broker_match(tol_abs: float, tol_bps: float) -> dict:
    data_dir = get_data_dir()
    trd_path = data_dir / "trades_raw.csv"
    bcf_path = data_dir / "broker_confirms.csv"
    if not trd_path.exists() or not bcf_path.exists():
        raise FileNotFoundError("Missing broker_confirms.csv or trades_raw.csv. Run Steps 1C and 3A first.")
    T = _pd.read_csv(trd_path)
    B = _pd.read_csv(bcf_path)

    def _norm_df(df: _pd.DataFrame, is_broker: bool) -> _pd.DataFrame:
        x = df.copy()
        for c in ["security_id", "side", "counterparty_legal_name"]:
            if c in x.columns:
                x[c] = x[c].astype(str).str.strip()
        if "security_id" in x.columns:
            x["security_id"] = x["security_id"].str.upper()
        if "side" in x.columns:
            x["side"] = x["side"].str.upper()
        if "counterparty_legal_name" in x.columns:
            x["counterparty_legal_name"] = x["counterparty_legal_name"].str.replace(r"\s+", " ", regex=True)
        if "trade_date" in x.columns:
            x["trade_date"] = _pd.to_datetime(x["trade_date"], errors="coerce").dt.date
        if "quantity" in x.columns:
            x["quantity"] = _pd.to_numeric(x["quantity"], errors="coerce").fillna(0).astype(int)
        if "price" in x.columns:
            x["price"] = _pd.to_numeric(x["price"], errors="coerce").astype(float)
        x["price_rounded_2"] = x["price"].round(2) if "price" in x.columns else _pd.NA
        def _mk_strict(r):
            return f"{r.get('trade_date')}|{r.get('side')}|{r.get('security_id')}|{r.get('quantity')}|{r.get('counterparty_legal_name')}|{r.get('price_rounded_2')}"
        def _mk_loose(r):
            return f"{r.get('trade_date')}|{r.get('side')}|{r.get('security_id')}|{r.get('quantity')}|{r.get('counterparty_legal_name')}"
        x["internal_key_strict" if not is_broker else "broker_key_strict"] = x.apply(_mk_strict, axis=1)
        x["internal_key_loose" if not is_broker else "broker_key_loose"] = x.apply(_mk_loose, axis=1)
        return x

    Tn = _norm_df(T, is_broker=False)
    Bn = _norm_df(B, is_broker=True)
    eligible_trades = len(Tn)
    loaded_confirms = len(Bn)

    exact = Tn.merge(Bn, left_on="internal_key_strict", right_on="broker_key_strict", suffixes=("_int", "_brk"))
    exact["affirmation_status"] = "Affirmed — exact"
    exact["price_diff"] = (exact["price_int"] - exact["price_brk"]).abs()
    exact["bps_diff"] = (exact["price_diff"] / exact["price_brk"].clip(lower=1e-9)) * 10000.0

    rem_T = Tn[~Tn["internal_key_strict"].isin(exact["internal_key_strict"])].copy()
    rem_B = Bn[~Bn["broker_key_strict"].isin(exact["broker_key_strict"])].copy()
    tol_cand = rem_T.merge(rem_B, left_on="internal_key_loose", right_on="broker_key_loose", suffixes=("_int", "_brk"))
    if not tol_cand.empty:
        tol_cand["price_diff"] = (tol_cand["price_int"] - tol_cand["price_brk"]).abs()
        tol_cand["bps_diff"] = (tol_cand["price_diff"] / tol_cand["price_brk"].clip(lower=1e-9)) * 10000.0
        within = tol_cand[(tol_cand["price_diff"] <= float(tol_abs)) | (tol_cand["bps_diff"] <= float(tol_bps))].copy()
        within["affirmation_status"] = "Affirmed — price within tolerance"
        out_tol = tol_cand.merge(within[["internal_key_loose", "broker_key_loose"]].drop_duplicates(), on=["internal_key_loose", "broker_key_loose"], how="left", indicator=True)
        tol_fail = out_tol[out_tol["_merge"] == "left_only"].drop(columns=["_merge"]).copy()
    else:
        within = _pd.DataFrame()
        tol_fail = _pd.DataFrame()

    affirmed = _pd.concat([exact, within], ignore_index=True, sort=False)

    mismatches = _pd.DataFrame()
    if not tol_fail.empty:
        tol_fail = tol_fail.copy()
        tol_fail["Mismatch Reason"] = "price_out_of_tolerance"
        mismatches = _pd.concat([mismatches, tol_fail], ignore_index=True, sort=False)

    diag_keys = ["trade_date", "security_id", "counterparty_legal_name"]
    if all(k in Tn.columns for k in diag_keys) and all(k in Bn.columns for k in diag_keys):
        diag_cand = rem_T.merge(rem_B, on=diag_keys, suffixes=("_int", "_brk"))
        if not diag_cand.empty:
            def _reason(r):
                if int(r.get("quantity_int", -1)) != int(r.get("quantity_brk", -1)):
                    return "quantity_mismatch"
                if str(r.get("side_int", "")).upper() != str(r.get("side_brk", "")).upper():
                    return "side_mismatch"
                if str(r.get("counterparty_legal_name", "")) == "":
                    return "counterparty_mismatch"
                if str(r.get("security_id", "")) == "":
                    return "security_mismatch"
                pdiff = abs(float(r.get("price_int", 0.0)) - float(r.get("price_brk", 0.0)))
                bpsd = (pdiff / max(float(r.get("price_brk", 0.0)), 1e-9)) * 10000.0
                if (pdiff <= float(tol_abs)) or (bpsd <= float(tol_bps)):
                    return ""
                return "price_out_of_tolerance"
            diag_cand["Mismatch Reason"] = diag_cand.apply(_reason, axis=1)
            diag_cand = diag_cand[diag_cand["Mismatch Reason"] != ""]
            if not diag_cand.empty:
                mismatches = _pd.concat([mismatches, diag_cand], ignore_index=True, sort=False)

    matched_T_keys = set(affirmed.get("internal_key_strict", _pd.Series(dtype=str)).tolist()) | set(affirmed.get("internal_key_loose", _pd.Series(dtype=str)).tolist())
    matched_B_keys = set(affirmed.get("broker_key_strict", _pd.Series(dtype=str)).tolist()) | set(affirmed.get("broker_key_loose", _pd.Series(dtype=str)).tolist())
    unmatched_internal = Tn[~Tn["internal_key_strict"].isin(matched_T_keys)].copy()
    unmatched_broker = Bn[~Bn["broker_key_strict"].isin(matched_B_keys)].copy()

    affirmed_count = len(affirmed)
    affirmed_pct = (affirmed_count / eligible_trades * 100.0) if eligible_trades else 0.0

    return {
        "affirmed": affirmed,
        "mismatches": mismatches,
        "unmatched_internal": unmatched_internal,
        "unmatched_broker": unmatched_broker,
        "eligible_trades": int(eligible_trades),
        "loaded_confirms": int(loaded_confirms),
        "affirmed_count": int(affirmed_count),
        "affirmed_pct": float(affirmed_pct),
    }
def _derive_lifecycle_status(df_in: _pd.DataFrame) -> _pd.DataFrame:
    """
    Vectorized lifecycle rules derived from settlement_date and break_flag.
    - If settlement_date <= today and break_flag == '🚨': 'Failed'
    - Else if settlement_date <= today: 'Settled'
    - Else: 'Pending'
    Returns a new DataFrame with a lifecycle_status column.
    """
    if df_in is None or df_in.empty:
        return df_in

    today = _date_local.today()
    out = []
    for _, r in df_in.iterrows():
        sd = None
        try:
            v = r.get("settlement_date")
            if isinstance(v, _date_local):
                sd = v
            elif v not in (None, ""):
                sd = _pd.to_datetime(v, errors="coerce").date()
        except Exception:
            sd = None
        b = r.get("break_flag")
        if sd and sd <= today and b == "🚨":
            out.append("Failed")
        elif sd and sd <= today:
            out.append("Settled")
        else:
            out.append("Pending")

    df_out = df_in.copy()
    df_out["lifecycle_status"] = out
    return df_out

def build_enriched_display_frame(df_numeric: _pd.DataFrame) -> _pd.DataFrame:
    if df_numeric is None or df_numeric.empty:
        return _pd.DataFrame()

    rename_map_disp = {
        "ticker": "Ticker",
        "company_name": "Company Name",
        "Website": "Company Website",
        "Sector": "Sector",
        "Industry": "Industry",
        "trade_date": "Trade Date",
        "settlement_date": "Settlement Date",
        "market_date": "Market Date",
        "unit_cost": "Unit Cost",
        "quantity": "Quantity",
        "market_price": "Market Price",
        "market_value": "Market Value",
        "cost_basis": "Cost Basis",
        "unrealized_gain_loss": "Unrealized Gain/Loss",
        "unrealized_gain_loss_pct": "Unrealized Gain/Loss %",
        "break_flag": "Break",
        "break_reason": "Break Reason",
        "break_reason_category": "Break Category",
        "lifecycle_status": "Lifecycle Status",
    }

    desired_order = [
        "Ticker",
        "Company Name",
        "Company Website",
        "Sector",
        "Industry",
        "Trade Date",
        "Settlement Date",
        "Market Date",
        "Quantity",
        "Unit Cost",
        "Cost Basis",
        "Market Price",
        "Day's Price $ Change",
        "Day's Price % Change",
        "Market Value",
        "Unrealized Gain/Loss",
        "Unrealized Gain/Loss %",
        "Day Value Change $",
        "Start Date Used",
        "End Date Used",
        "Lifecycle Status",
        "Break Category",
        "Break Reason",
        "Break",
    ]

    df_show = df_numeric.copy()
    safe_map = {k: v for k, v in rename_map_disp.items() if k in df_show.columns}
    df_disp = df_show.rename(columns=safe_map)

    if "Unit Cost" in df_disp.columns and "unit_cost_num" in df_show.columns:
        df_disp["Unit Cost"] = df_show["unit_cost_num"].apply(_fmt_money)
    if "Quantity" in df_disp.columns and "quantity_num" in df_show.columns:
        df_disp["Quantity"] = df_show["quantity_num"].apply(_fmt_qty)
    if "Market Price" in df_disp.columns and "market_price_num" in df_show.columns:
        df_disp["Market Price"] = df_show["market_price_num"].apply(_fmt_money)
    if "Market Value" in df_disp.columns and "market_value" in df_show.columns:
        df_disp["Market Value"] = df_show["market_value"].apply(_fmt_money)
    if "Cost Basis" in df_disp.columns and "cost_basis" in df_show.columns:
        df_disp["Cost Basis"] = df_show["cost_basis"].apply(_fmt_money)
    if "Unrealized Gain/Loss" in df_disp.columns and "unrealized_gain_loss" in df_show.columns:
        df_disp["Unrealized Gain/Loss"] = df_show["unrealized_gain_loss"].apply(_fmt_money_signed)
    if "Unrealized Gain/Loss %" in df_disp.columns and "unrealized_gain_loss_pct" in df_show.columns:
        df_disp["Unrealized Gain/Loss %"] = df_show["unrealized_gain_loss_pct"].apply(_fmt_pct_signed)

    if "Day Value Change $" in df_show.columns:
        df_disp["Day Value Change $"] = df_show["Day Value Change $"] .apply(_fmt_money_signed)
    if "Day's Price $ Change" in df_show.columns:
        df_disp["Day's Price $ Change"] = df_show["Day's Price $ Change"].apply(_fmt_money_signed)
    if "Day's Price % Change" in df_show.columns:
        df_disp["Day's Price % Change"] = df_show["Day's Price % Change"].apply(_fmt_pct_signed)

    # Normalize Market Date display format to yyyy-mm-dd without touching numeric/base frames
    if "Market Date" in df_disp.columns:
        try:
            md_series = _pd.to_datetime(df_disp["Market Date"], errors="coerce")
            df_disp["Market Date"] = md_series.dt.strftime('%Y-%m-%d')
        except Exception:
            pass

    final_cols = [c for c in desired_order if c in df_disp.columns]
    return df_disp[final_cols]

# ======================== Break helpers and KPI tiles ========================
def apply_breaks(df_numeric: _pd.DataFrame, abs_thr: float, pct_thr: float) -> _pd.DataFrame:
    """
    Compute break flags using absolute $ and % thresholds.
    Adds:
      - break_abs_delta, break_pct_delta
      - break_flag (✅, 🚨, ⚠️)
      - break_reason (text)
      - break_reason_category: "Threshold Breach", "No Price Data", "Data Missing", or "OK"
    """
    if df_numeric is None or df_numeric.empty:
        return df_numeric

    df = df_numeric.copy()
    # Ensure clean unrealized_gain_loss_pct
    try:
        df = _safe_ugl_pct(df)
    except Exception:
        pass

    qty = _pd.to_numeric(df.get("quantity_num"), errors="coerce")
    unit_cost = _pd.to_numeric(df.get("unit_cost_num"), errors="coerce")
    price = _pd.to_numeric(
        df.get("market_price_num") if "market_price_num" in df.columns else df.get("market_price"),
        errors="coerce",
    )
    cost_basis = _pd.to_numeric(df.get("cost_basis"), errors="coerce")
    abs_delta = _pd.to_numeric(df.get("unrealized_gain_loss"), errors="coerce")
    pct_delta = _pd.to_numeric(df.get("unrealized_gain_loss_pct"), errors="coerce")
    if "unrealized_gain_loss_pct" not in df.columns or pct_delta.isna().all():
        pct_delta = _np.where(
            (cost_basis > 0) & (~abs_delta.isna()),
            (abs_delta / cost_basis) * 100.0,
            _np.nan,
        )

    flags, reasons, cats = [], [], []
    for i in range(len(df)):
        a = abs_delta.iloc[i] if not _pd.isna(abs_delta.iloc[i]) else _np.nan
        p = pct_delta[i] if isinstance(pct_delta, _np.ndarray) else pct_delta.iloc[i]
        q = qty.iloc[i] if len(qty) > i else _np.nan
        uc = unit_cost.iloc[i] if len(unit_cost) > i else _np.nan
        pr = price.iloc[i] if len(price) > i else _np.nan

        # Data-quality checks first
        if _pd.isna(q) or _pd.isna(uc):
            flags.append("⚠️"); reasons.append("Missing quantity or unit cost"); cats.append("Data Missing"); continue
        if _pd.isna(pr):
            flags.append("🚨"); reasons.append("No market price available"); cats.append("No Price Data"); continue
        if _pd.isna(a) or _pd.isna(p):
            flags.append("⚠️"); reasons.append("Missing P/L values"); cats.append("Data Missing"); continue

        # Threshold evaluation
        if abs(a) > float(abs_thr) or abs(p) > float(pct_thr):
            flags.append("🚨"); reasons.append(f"Abs {a:+,.2f} vs {abs_thr:,.0f} or Pct {p:+.2f}% vs {pct_thr:.2f}%"); cats.append("Threshold Breach")
        else:
            flags.append("✅"); reasons.append("Within thresholds"); cats.append("OK")

    df["break_abs_delta"] = abs_delta
    df["break_pct_delta"] = pct_delta
    df["break_flag"] = flags
    df["break_reason"] = reasons
    df["break_reason_category"] = cats
    return df

def _render_kpis(df_num: _pd.DataFrame):
    if df_num is None or df_num.empty:
        return
    mv = _pd.to_numeric(df_num.get("market_value"), errors="coerce").replace([_np.inf, -_np.inf], _np.nan)
    u = _pd.to_numeric(df_num.get("unrealized_gain_loss"), errors="coerce").replace([_np.inf, -_np.inf], _np.nan)
    pct = _pd.to_numeric(df_num.get("unrealized_gain_loss_pct"), errors="coerce").replace([_np.inf, -_np.inf], _np.nan)
    breaks = (df_num.get("break_flag") == "🚨").sum() if "break_flag" in df_num.columns else 0
    total_pos = len(df_num)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Total Portfolio Value", f"{_np.nansum(mv):,.2f}")
    with col2: st.metric("Total Unrealized P/L", f"{_np.nansum(u):,.2f}")
    with col3:
        finite_pct = pct[_np.isfinite(pct)] if hasattr(pct, "__array__") else pct
        avg_pct = float(_np.nanmean(finite_pct)) if hasattr(finite_pct, "__array__") and finite_pct.size else 0.0
        st.metric("Avg Unrealized %", f"{avg_pct:.2f}%")
    with col4: st.metric("Breaks", f"{int(breaks)}")
    with col5: st.metric("Break Rate", f"{(breaks/total_pos*100 if total_pos else 0):.1f}%")
# UI skeleton — no file logic yet

## page_config moved to top


def render_footer() -> None:
    """Render the footer with dynamic month and year.

    UI skeleton — no file logic yet
    """
    date_str = datetime.now().strftime("%B %Y")
    st.caption(f"App Version 1.0 | Last updated {date_str}")


def page_upload_review() -> None:
    """Upload & Review page.

    Flexible file upload and user-driven column mapping for robust onboarding.
    """
    st.header("Upload & Review")
    st.write(
        "Upload internal positions file for review and basic summaries."
    )
    
    # Sample file generation for user onboarding
    st.write("Download a sample positions template to see required columns and format.")
    
    # Generate sample data
    import pandas as pd
    from io import BytesIO
    
    sample_data = {
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
        'quantity': [100, 50, 75, 200, 25],
        'unit_cost': [150.25, 300.50, 2800.75, 250.00, 3500.25],
        'trade_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
        'settlement_date': ['2024-01-17', '2024-01-18', '2024-01-19', '2024-01-22', '2024-01-23'],
        'status': ['Settled', 'Pending', 'Settled', 'Pending', 'Settled'],
        'sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 'Consumer Discretionary'],
        'industry': ['Consumer Electronics', 'Software', 'Internet Services', 'Automotive', 'E-commerce'],
        'counterparty': ['Goldman Sachs', 'Morgan Stanley', 'JPMorgan', 'Citigroup', 'Bank of America'],
        'trade_type': ['Buy', 'Sell', 'Buy', 'Buy', 'Sell']
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_sample.to_excel(writer, sheet_name='Positions', index=False)
    
    output.seek(0)
    
    # Download button
    st.download_button(
        label="Download Sample File",
        data=output.getvalue(),
        file_name="sample_positions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download a sample Excel file with position data columns"
    )
    
    st.markdown("---")
    
    # Flexible file upload and user-driven column mapping for robust onboarding
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        help="Upload CSV or Excel files for reconciliation"
    )
    
    if uploaded_file is not None:
        # --- Auto-dismiss success/info after file upload ---
        msg_placeholder = st.empty()
        name_placeholder = st.empty()
        msg_placeholder.success("File uploaded! Ready to process.")
        name_placeholder.info(f"Filename: {uploaded_file.name}")
        import time as _time
        _time.sleep(3)
        msg_placeholder.empty()
        name_placeholder.empty()
        # session_state caches data across tabs; cleared on new upload
        if st.session_state.get("current_upload_name") != uploaded_file.name:
            st.session_state["current_upload_name"] = uploaded_file.name
            for key in ["alpha_cache", "df_enriched_base", "alpha_fetch_complete"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Reset mapping completion state for a new file
            st.session_state["mapping_complete"] = False

        # Read uploaded file into DataFrame
        try:
            name_lower = uploaded_file.name.lower()
            uploaded_file.seek(0)
            if name_lower.endswith(".csv"):
                df_uploaded = pd.read_csv(uploaded_file)
            elif name_lower.endswith((".xlsx", ".xls")):
                df_uploaded = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return
        except Exception as read_error:
            st.error(f"Could not read the uploaded file: {read_error}")
            return

        # Remove index-like columns before any preview or mapping
        # Drops columns with blank names or names starting with 'Unnamed:'
        columns_to_drop = []
        for uploaded_column in list(df_uploaded.columns):
            column_name_as_string = "" if uploaded_column is None else str(uploaded_column)
            normalized_column_name = column_name_as_string.strip()
            if normalized_column_name == "" or normalized_column_name.startswith("Unnamed:"):
                columns_to_drop.append(uploaded_column)
        if columns_to_drop:
            df_uploaded = df_uploaded.drop(columns=columns_to_drop)

        # Normalize index to avoid showing a saved index as the first column in the UI
        df_uploaded = df_uploaded.reset_index(drop=True)

        # Show basic info and preview
        num_rows, num_cols = df_uploaded.shape
        st.write(f"Rows: {num_rows} | Columns: {num_cols}")
        st.dataframe(df_uploaded.head(10), use_container_width=True, hide_index=True)

        # Required columns UI — hide once enriched positions exist
        try:
            import pandas as _pd
        except Exception:
            pass
        _has_enriched_positions = isinstance(st.session_state.get("df_enriched_display"), _pd.DataFrame) and bool(st.session_state.get("df_enriched_display") is not None and not st.session_state.get("df_enriched_display").empty)

        if not _has_enriched_positions:
            if "mapping_complete" not in st.session_state:
                st.session_state["mapping_complete"] = False

            if not st.session_state.get("mapping_complete", False):
                # Expected columns and user-selected required columns
                expected_columns = [
                    "ticker",
                    "quantity",
                    "unit_cost",
                    "trade_date",
                    "settlement_date",
                    "status",
                    "sector",
                    "industry",
                    "counterparty",
                    "trade_type",
                ]

                st.markdown("Select required columns for your use case:")
                default_required = ["ticker", "quantity", "unit_cost"]
                selected_required = st.multiselect(
                    "Required columns",
                    options=expected_columns,
                    default=default_required,
                    key="required_columns_selection",
                    help="Choose which fields must be present before proceeding",
                )

                # Presence check (case-insensitive)
                dataframe_columns = list(df_uploaded.columns)
                lower_to_original = {c.lower(): c for c in dataframe_columns}
                missing_required = [
                    req for req in selected_required if req.lower() not in lower_to_original
                ]

                if missing_required:
                    st.warning(
                        "Some required columns are missing from your file. Please map them below."
                    )

                    with st.expander("Map required columns to your file headers", expanded=True):
                        mapping_selections = {}
                        options_for_select = ["Select"] + dataframe_columns

                        for required_field in selected_required:
                            # Pre-fill using case-insensitive match
                            prefill_match = lower_to_original.get(required_field.lower())
                            if prefill_match and prefill_match in dataframe_columns:
                                default_index = options_for_select.index(prefill_match)
                            else:
                                default_index = 0

                            selected_header = st.selectbox(
                                f"Map '{required_field}' to:",
                                options=options_for_select,
                                index=default_index,
                                key=f"map_select_{required_field}",
                            )
                            if selected_header != "Select":
                                mapping_selections[required_field] = selected_header

                        apply_mapping_clicked = st.button("Apply mapping", key="apply_mapping_button")

                        if apply_mapping_clicked:
                            # Build rename map from chosen headers to required field names
                            rename_map = {
                                chosen_header: required
                                for required, chosen_header in mapping_selections.items()
                            }

                            if rename_map:
                                df_uploaded = df_uploaded.rename(columns=rename_map)

                            # Re-check required columns after renaming
                            still_missing = [
                                req for req in selected_required if req not in df_uploaded.columns
                            ]

                            if still_missing:
                                st.error(
                                    "Missing required columns after mapping: "
                                    + ", ".join(still_missing)
                                )
                            else:
                                st.success("Mapping applied. Preview updated.")
                                df_uploaded = df_uploaded.reset_index(drop=True)
                                st.session_state["uploaded_mapped_df"] = df_uploaded
                                st.session_state["mapping_complete"] = True
                                st.dataframe(df_uploaded.head(10), use_container_width=True, hide_index=True)
                else:
                    st.success("All required columns are present.")
                    df_uploaded = df_uploaded.reset_index(drop=True)
                    st.session_state["uploaded_mapped_df"] = df_uploaded
                    st.session_state["mapping_complete"] = True

        # ==============================================
        # Intraday→fallback: GLOBAL_QUOTE then TIME_SERIES_DAILY_ADJUSTED
        # Dual-endpoint Alpha Vantage calls for price and metadata.
        # Rate limit caution: 75 calls/minute premium plan; throttle established.
        # No fallback to unit cost; failures flagged.
        # session_state caches data across tabs; cleared on new upload
        # ==============================================

        # enrichment is now imported from alpha_vantage module

        # Run enrichment if not yet cached; otherwise reuse cache
        import pandas as pd
        if not st.session_state.get("alpha_fetch_complete"):
            df_mapped_input = st.session_state.get("uploaded_mapped_df")
            # Safety guard: ensure required columns exist before enrichment
            if not isinstance(df_mapped_input, pd.DataFrame):
                st.info("Please upload a file and complete required column mapping before enrichment.")
                return
            _required_for_enrich = ["ticker", "quantity", "unit_cost"]
            _missing_for_enrich = [c for c in _required_for_enrich if c not in df_mapped_input.columns]
            if _missing_for_enrich:
                st.error("Missing required columns for enrichment: " + ", ".join(_missing_for_enrich) + ". Please map them above.")
                return
            df_base = enrich_with_alpha_vantage(df_mapped_input, ENTITLEMENT_MODE)
        else:
            # Show the cached DataFrame whenever the user re-opens Upload & Review without re-calling any API.
            df_base = st.session_state.get("df_enriched_base")

        if df_base is not None:
            abs_thr = float(st.session_state.get("abs_threshold", 1000.0))
            pct_thr = float(st.session_state.get("pct_threshold", 5.0))

            def compute_flag(row) -> str:
                import pandas as _pd
                try:
                    abs_val = float(row.get("unrealized_gain_loss"))
                    pct_val = float(row.get("unrealized_gain_loss_pct"))
                except Exception:
                    return "⚠️"
                if _pd.isna(abs_val) or _pd.isna(pct_val):
                    return "⚠️"
                if abs(abs_val) > abs_thr or abs(pct_val) > pct_thr:
                    return "🚨"
                return "✅"

            df_show = df_base.copy()
            df_show["break_flag"] = df_show.apply(compute_flag, axis=1)

            # Display table in required order with formatting
            # Number & percent formatting via Streamlit column_config / pandas style
            rename_map_disp = {
                "ticker": "Ticker",
                "company_name": "Company Name",
                "Website": "Company Website",
                "Sector": "Sector",
                "Industry": "Industry",
                "trade_date": "Trade Date",
                "settlement_date": "Settlement Date",
                "market_date": "Market Date",
                "unit_cost": "Unit Cost",
                "quantity": "Quantity",
                "market_price": "Market Price",
                "market_value": "Market Value",
                "cost_basis": "Cost Basis",
                "unrealized_gain_loss": "Unrealized Gain/Loss",
                "unrealized_gain_loss_pct": "Unrealized Gain/Loss %",
                "break_flag": "Break",
            }
            desired_order = [
                "Ticker",
                "Company Name",
                "Company Website",
                "Sector",
                "Industry",
                "Trade Date",
                "Settlement Date",
                "Market Date",
                "Unit Cost",
                "Quantity",
                "Market Price",
                "Market Value",
                "Cost Basis",
                "Unrealized Gain/Loss",
                "Unrealized Gain/Loss %",
                "Break",
            ]

            df_disp = df_show.rename(columns=rename_map_disp)
            df_disp = df_disp[[c for c in desired_order if c in df_disp.columns]]

            # Build a formatted view to guarantee exact formatting regardless of column_config rendering

            df_fmt = df_disp.copy()
            if "Unit Cost" in df_fmt.columns:
                df_fmt["Unit Cost"] = df_show["unit_cost_num"].apply(_fmt_money)
            if "Quantity" in df_fmt.columns:
                df_fmt["Quantity"] = df_show["quantity_num"].apply(_fmt_qty)
            if "Market Price" in df_fmt.columns:
                df_fmt["Market Price"] = df_show["market_price_num"].apply(_fmt_money)
            if "Market Value" in df_fmt.columns and "market_value" in df_show.columns:
                df_fmt["Market Value"] = df_show["market_value"].apply(_fmt_money)
            if "Cost Basis" in df_fmt.columns:
                df_fmt["Cost Basis"] = df_show["cost_basis"].apply(_fmt_money)
            if "Unrealized Gain/Loss" in df_fmt.columns:
                df_fmt["Unrealized Gain/Loss"] = df_show["unrealized_gain_loss"].apply(_fmt_money_signed)
            if "Unrealized Gain/Loss %" in df_fmt.columns:
                df_fmt["Unrealized Gain/Loss %"] = df_show["unrealized_gain_loss_pct"].apply(_fmt_pct_signed)

            # Data cached in session_state; persists across tab navigation.
            st.session_state["df_enriched_base"] = df_base  # persist raw numbers

            # Summary removed; KPI tiles are rendered later above the final table

        # =============== Day Over Day Computation from cache only ===============
        if "df_enriched_base" in st.session_state and st.session_state["df_enriched_base"] is not None:
            df_base_local = st.session_state["df_enriched_base"]
        else:
            df_base_local = None

        # ================= Calendar-aware Day Over Day helpers =================
        from datetime import datetime, date
        from zoneinfo import ZoneInfo

        # Optional calendar dependency with safe fallback
        try:
            import pandas_market_calendars as mcal
            _XNYS = mcal.get_calendar("XNYS")
        except Exception:
            _XNYS = None  # fallback path will use Alpha Vantage daily keys

        _TZ_NY = ZoneInfo("America/New_York")

        def _ny_now():
            return datetime.now(_TZ_NY)

        def _trading_day_on_or_before(target: date) -> date | None:
            """
            Return the last valid NYSE trading day on or before target.
            Uses pandas_market_calendars when available; otherwise inspects any cached
            Alpha Vantage daily series to find the most recent <= target.
            """
            # Preferred: exchange calendar (handles holidays/early closes)
            if _XNYS is not None:
                start = target - _timedelta(days=14)
                sched = _XNYS.schedule(start_date=start, end_date=target)
                if not sched.empty:
                    return sched.index[-1].date()
                return None

            # Fallback: use any non-empty daily series in cache to infer valid dates
            daily_cache: dict = st.session_state.get("daily_cache", {})
            for ts in daily_cache.values():
                if isinstance(ts, dict) and ts:
                    keys = sorted(ts.keys())  # 'YYYY-MM-DD' lexicographic works
                    tstr = target.strftime("%Y-%m-%d")
                    last = None
                    for k in keys:
                        if k <= tstr:
                            last = k
                        else:
                            break
                    if last:
                        return datetime.strptime(last, "%Y-%m-%d").date()
            return None

        def _previous_trading_day(end_trading_day: date, n: int = 1) -> date | None:
            """
            Get the n-th previous trading day before the given trading day.
            """
            if _XNYS is not None:
                start = end_trading_day - _timedelta(days=40)
                sched = _XNYS.schedule(start_date=start, end_date=end_trading_day)
                if len(sched.index) >= n + 1:
                    return sched.index[-(n + 1)].date()
                return None

            # Fallback using cached Alpha Vantage keys
            daily_cache: dict = st.session_state.get("daily_cache", {})
            for ts in daily_cache.values():
                if isinstance(ts, dict) and ts:
                    keys = sorted(ts.keys())
                    end_str = end_trading_day.strftime("%Y-%m-%d")
                    if end_str in keys:
                        idx = keys.index(end_str)
                        if idx - n >= 0:
                            return datetime.strptime(keys[idx - n], "%Y-%m-%d").date()
                        return None
                    prev = [k for k in keys if k <= end_str]
                    if len(prev) >= n + 1:
                        return datetime.strptime(prev[-(n + 1)], "%Y-%m-%d").date()
            return None

        def _daily_close_on_or_before(ts_daily: dict, target: date) -> tuple[str | None, float | None]:
            """
            Given an Alpha Vantage daily dict and a target date, return (date_str, close_float)
            for the most recent trading day on or before target.
            """
            if not isinstance(ts_daily, dict) or not ts_daily:
                return None, None
            tstr = target.strftime("%Y-%m-%d")
            for d in sorted(ts_daily.keys(), reverse=True):
                if d <= tstr:
                    row = ts_daily.get(d, {})
                    val = row.get("4. close") or row.get("5. adjusted close")
                    try:
                        return d, float(val)
                    except Exception:
                        return d, None
            return None, None

        def _latest_intraday_price(sym: str) -> tuple[str | None, float | None]:
            """
            Use cached TIME_SERIES_INTRADAY to get the most recent bar's close for sym.
            Returns (timestamp_str, price) or (None, None).
            """
            intraday_cache: dict = st.session_state.get("intraday_cache", {})
            ts = intraday_cache.get(sym)
            if isinstance(ts, dict) and ts:
                try:
                    ts_key = max(ts.keys())
                    val = ts[ts_key].get("4. close")
                    return ts_key, (float(val) if val not in (None, "") else None)
                except Exception:
                    return None, None
            return None, None

        def _market_open_now_ny() -> bool:
            """
            True if right now is within the regular XNYS session.
            Falls back to a simple time check if calendar is not available.
            """
            now = _ny_now()
            if _XNYS is not None:
                sched = _XNYS.schedule(start_date=now.date(), end_date=now.date())
                if sched.empty:
                    return False
                open_dt = sched.iloc[0]["market_open"].tz_convert(_TZ_NY).to_pydatetime()
                close_dt = sched.iloc[0]["market_close"].tz_convert(_TZ_NY).to_pydatetime()
                return open_dt <= now <= close_dt

            # Basic fallback: 9:30-16:00 ET, Mon-Fri
            if now.weekday() > 4:
                return False
            minutes = now.hour * 60 + now.minute
            return (9 * 60 + 30) <= minutes <= (16 * 60)

        def _recompute_day_over_day_from_cache(df_base_input):
            """
            Recompute DoD strictly from selected Start/End trading days.
            Only updates these columns in df_enriched_base:
              - Start Date Used
              - End Date Used
              - Day Value Change $
              - Day Value Change %
            Leaves all other columns/behavior untouched.
            """
            if df_base_input is None:
                return

            import numpy as _np
            import pandas as _pd
            from datetime import date as _date_local

            df = df_base_input.copy()
            daily_cache: dict = st.session_state.get("daily_cache", {})

            # Sidebar params
            use_custom = bool(st.session_state.get("use_custom_dod"))
            start_param: _date_local = st.session_state.get("dod_start_date")
            end_param: _date_local = st.session_state.get("dod_end_date")

            # Ensure daily cache has the series we need for all tickers
            # (Lazy backfill; preserves existing caches and UI behavior.)
            tickers = (
                df.get("__ticker_norm__", _pd.Series([], dtype=str))
                .astype(str)
                .str.upper()
                .dropna()
                .unique()
                .tolist()
            )
            missing_daily = [t for t in tickers if not isinstance(daily_cache.get(t), dict) or not daily_cache.get(t)]
            if missing_daily:
                import os as _os
                import time as _time
                import requests as _requests
                from dotenv import load_dotenv as _load_dotenv

                _load_dotenv()
                api_key = _os.getenv("ALPHA_VANTAGE_KEY")
                if not api_key:
                    st.warning("ALPHA_VANTAGE_KEY is not set. Day-over-day backfill skipped.")
                else:
                    session = _requests.Session()
                    window_start = _time.monotonic()
                    calls_in_window = 0

                    def _guard():
                        nonlocal window_start, calls_in_window
                        now = _time.monotonic()
                        elapsed = now - window_start
                        if elapsed >= 60:
                            window_start = now
                            calls_in_window = 0
                        if calls_in_window >= 70:
                            sleep_s = max(0.0, 60 - elapsed)
                            st.info(f"Pausing {sleep_s:.0f}s to respect API rate limits…")
                            _time.sleep(sleep_s)
                            window_start = _time.monotonic()
                            calls_in_window = 0

                    prog = st.progress(0)
                    total_miss = max(1, len(missing_daily))
                    for idx, sym in enumerate(missing_daily, start=1):
                        _guard()
                        try:
                            resp = session.get(
                                "https://www.alphavantage.co/query",
                                params={
                                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                                    "symbol": sym,
                                    "outputsize": "compact",
                                    "apikey": api_key,
                                    "entitlement": ENTITLEMENT_MODE,
                                },
                                timeout=20,
                            )
                            calls_in_window += 1
                            data = resp.json() if resp.ok else {}
                            ts = data.get("Time Series (Daily)", {})
                            if isinstance(ts, dict) and ts:
                                daily_cache[sym] = ts
                        except Exception:
                            pass
                        prog.progress(int(idx / total_miss * 100))
                        _guard()
                    prog.empty()
                    st.session_state["daily_cache"] = daily_cache

            # Resolve End trading day
            resolved_end = _trading_day_on_or_before(end_param if (use_custom and isinstance(end_param, _date_local)) else _date_local.today())

            # Resolve Start trading day
            if use_custom and isinstance(start_param, _date_local):
                resolved_start = _trading_day_on_or_before(start_param)
            else:
                resolved_start = _previous_trading_day(resolved_end) if resolved_end else None

            # Decide if intraday is allowed for End (only when End is today AND market open)
            use_intraday_end = bool(resolved_end and (resolved_end == _date_local.today()) and _market_open_now_ny())

            # Local helpers (do NOT change globals)
            def _daily_close_on_or_before(ts_daily: dict, target: _date_local):
                """Return (YYYY-MM-DD, close_float) for most recent trading day <= target using cached AV daily."""
                if not isinstance(ts_daily, dict) or not ts_daily:
                    return None, None
                tstr = target.strftime("%Y-%m-%d")
                for d in sorted(ts_daily.keys(), reverse=True):
                    if d <= tstr:
                        row = ts_daily.get(d, {})
                        val = row.get("4. close") or row.get("5. adjusted close")
                        try:
                            return d, float(val)
                        except Exception:
                            return d, None
                return None, None

            def _latest_intraday_price(sym: str):
                """Use cached TIME_SERIES_INTRADAY to get the latest bar's close for sym. Returns (ts_str, float) or (None, None)."""
                ts_map = st.session_state.get("intraday_cache", {}).get(sym)
                if isinstance(ts_map, dict) and ts_map:
                    try:
                        k = max(ts_map.keys())
                        v = ts_map[k].get("4. close")
                        return k, (float(v) if v not in (None, "") else None)
                    except Exception:
                        return None, None
                return None, None

            start_used, end_used, dv_list, dvpct_list = [], [], [], []

            for _, r in df.iterrows():
                sym = str(r.get("__ticker_norm__", "")).upper()
                qty = _pd.to_numeric(r.get("quantity_num"), errors="coerce")
                ts_daily = daily_cache.get(sym)

                # End price resolution
                end_label, p_end = None, _np.nan
                if resolved_end is not None:
                    if use_intraday_end:
                        ts_str, intraday_p = _latest_intraday_price(sym)
                        if intraday_p is not None:
                            end_label, p_end = ts_str, intraday_p
                        else:
                            end_label, p_end = _daily_close_on_or_before(ts_daily, resolved_end)
                    else:
                        end_label, p_end = _daily_close_on_or_before(ts_daily, resolved_end)

                # Start price resolution
                start_label, p_start = None, _np.nan
                if resolved_start is not None:
                    start_label, p_start = _daily_close_on_or_before(ts_daily, resolved_start)

                # Compute DoD as total position value change (qty * per-share change)
                if _pd.isna(p_end) or _pd.isna(p_start):
                    dv = _np.nan
                    dp = _np.nan
                else:
                    try:
                        q_float = float(qty) if not _pd.isna(qty) else _np.nan
                    except Exception:
                        q_float = _np.nan

                    if _pd.isna(q_float):
                        dv = _np.nan
                        dp = _np.nan
                    else:
                        per_share_change = float(p_end) - float(p_start)
                        prior_value = q_float * float(p_start)
                        current_value = q_float * float(p_end)
                        dv = current_value - prior_value
                        dp = (_np.nan if prior_value in (None, 0) else (dv / prior_value) * 100.0)

                start_used.append(start_label)
                end_used.append(end_label)
                dv_list.append(dv)
                dvpct_list.append(dp)

            # Only update the intended columns
            df["Start Date Used"] = start_used
            df["End Date Used"] = end_used
            df["Day Value Change $"] = dv_list
            df["Day Value Change %"] = dvpct_list

            # Persist numeric frame and rebuild display using existing helpers (unchanged behavior)
            st.session_state["df_enriched_base"] = df
            abs_thr = float(st.session_state.get("abs_threshold", 1000.0))
            pct_thr = float(st.session_state.get("pct_threshold", 5.0))
            _df_with_breaks = apply_breaks(df, abs_thr, pct_thr)
            st.session_state["df_enriched_base"] = _df_with_breaks
            st.session_state["df_enriched_display"] = build_enriched_display_frame(_df_with_breaks)

        # === Trigger block (unchanged intent; safe against NameError) ===
        need_recompute = False
        if df_base_local is not None:
            try:
                import pandas as _pd
                base_missing = (
                    not isinstance(df_base_local, _pd.DataFrame)
                    or ("Day Value Change $" not in df_base_local.columns)
                    or ("Day Value Change %" not in df_base_local.columns)
                )
                disp = st.session_state.get("df_enriched_display")
                disp_missing = (
                    not isinstance(disp, _pd.DataFrame)
                    or ("Day Value Change $" not in disp.columns)
                    or ("Day Value Change %" not in disp.columns)
                )
                _apply_flag = bool(globals().get("apply_dod", False))
                need_recompute = _apply_flag or base_missing or disp_missing or bool(st.session_state.get("force_recompute_dod"))
            except Exception:
                need_recompute = True

        if df_base_local is not None and need_recompute:
            _recompute_day_over_day_from_cache(df_base_local)
            st.session_state["force_recompute_dod"] = False

        # Final render moved below, outside the upload block, to persist across tab switches

        

    # Always-on final render from cache only; no API calls
    import pandas as _pd
    if "df_enriched_base" in st.session_state and isinstance(st.session_state["df_enriched_base"], _pd.DataFrame):
        if "df_enriched_display" not in st.session_state or not isinstance(st.session_state["df_enriched_display"], _pd.DataFrame):
            st.session_state["df_enriched_display"] = build_enriched_display_frame(st.session_state["df_enriched_base"])

    if "df_enriched_display" in st.session_state and isinstance(st.session_state["df_enriched_display"], _pd.DataFrame) and not st.session_state["df_enriched_display"].empty:
        # Apply breaks again on each render from current thresholds to stay in sync
        abs_thr = float(st.session_state.get("abs_threshold", 1000.0))
        pct_thr = float(st.session_state.get("pct_threshold", 5.0))
        _df_num = st.session_state.get("df_enriched_base")
        if isinstance(_df_num, _pd.DataFrame) and not _df_num.empty:
            _df_num = apply_breaks(_df_num, abs_thr, pct_thr)
            st.session_state["df_enriched_base"] = _df_num
            st.session_state["df_enriched_display"] = build_enriched_display_frame(_df_num)

        # --- Insert KPI section right before rendering Enriched Positions ---
        _df_num = st.session_state.get("df_enriched_base")
        if isinstance(_df_num, _pd.DataFrame) and not _df_num.empty:
            kpi_container = st.container()
            with kpi_container:
                # Scoped CSS and wrapper for vertical rhythm
                st.markdown(
                    """
                    <style>
                      .kpi-scope { margin-bottom: 12px; }
                      .kpi-scope .kpi-wrap { text-align: center; }
                      .kpi-scope .kpi-title { 
                        font-size: 0.95rem; 
                        color: #6b7280; 
                         font-weight: 400; 
                        margin-bottom: 0.25rem; 
                        line-height: 1.2;
                      }
                      .kpi-scope .kpi-value { 
                        font-size: 1.75rem; 
                        font-weight: 800; 
                        line-height: 1.1; 
                        margin-top: 0.05rem;
                        display: block;
                        text-align: center;
                      }
                      /* Ensure centering even if columns are not nested under .kpi-scope */
                      .kpi-wrap { text-align: center; }
                      .kpi-title { font-weight: 400; }
                      .kpi-value { display: block; text-align: center; font-weight: 800; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                # Final override to ensure KPI values are 1.75rem and centered
                st.markdown(
                    """
                    <style>
                      .kpi-value {
                        font-size: 1.75rem !important;
                        font-weight: 800;
                        display: block;
                        text-align: center;
                      }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="kpi-scope">', unsafe_allow_html=True)
                # Right-aligned refresh timestamp (ET)
                try:
                    from zoneinfo import ZoneInfo as _ZoneInfo
                    _tz_ny = _ZoneInfo("America/New_York")
                    _ts_et = st.session_state.get("alpha_last_refresh_et")
                    if _ts_et is None:
                        _ts_et = _pd.Timestamp.now(tz=_tz_ny).to_pydatetime()
                    _ts_str = _pd.Timestamp(_ts_et).tz_convert(_tz_ny).strftime("%m/%d/%Y %I:%M %p")
                    st.markdown(
                        f"<div style='text-align:right; font-size:0.9rem; color:#6b7280;'>As of {_ts_str} ET<br>15-minute delayed</div>",
                        unsafe_allow_html=True,
                    )
                    # Add a bit of vertical space below the timestamp block
                    st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)
                except Exception:
                    pass
                mv   = _pd.to_numeric(_df_num.get("market_value"),              errors="coerce")
                u    = _pd.to_numeric(_df_num.get("unrealized_gain_loss"),      errors="coerce")
                cb   = _pd.to_numeric(_df_num.get("cost_basis"),                errors="coerce")
                dayv = _pd.to_numeric(_df_num.get("Day Value Change $"),          errors="coerce")

                mv_sum = _np.nansum(mv)
                u_sum  = _np.nansum(u)
                cb_sum = _np.nansum(cb)

                # Portfolio Unrealized % = total UGL / total Cost Basis
                port_unreal_pct = (_np.nan if cb_sum <= 0 else (u_sum / cb_sum * 100.0))

                # Portfolio Day's Value stats
                day_val_sum = _np.nansum(dayv) if isinstance(dayv, _pd.Series) else _np.nan
                if isinstance(dayv, _pd.Series):
                    prev_val_rowwise = mv.subtract(dayv)
                    prev_val_sum = _np.nansum(prev_val_rowwise)
                    day_pct_port = (_np.nan if (prev_val_sum is None or prev_val_sum <= 0) else (day_val_sum / prev_val_sum * 100.0))
                else:
                    day_pct_port = _np.nan

                # Breaks as before
                breaks = int((_df_num.get("break_flag") == "🚨").sum()) if "break_flag" in _df_num.columns else 0
                total_pos = len(_df_num)
                break_rate = (breaks / total_pos * 100) if total_pos else 0

                # Existing color logic retained for signed metrics
                def _color_text(val_float: float, is_pct: bool = False) -> str:
                    if _pd.isna(val_float):
                        return '<div class="kpi-wrap"><span class="kpi-value"></span></div>'
                    color = "#137333" if float(val_float) > 0 else ("#b3261e" if float(val_float) < 0 else "inherit")
                    txt = _fmt_pct_signed(val_float) if is_pct else _fmt_money_signed(val_float)
                    return f'<div class=\"kpi-wrap\"><span class=\"kpi-value\" style=\"color:{color};\">{txt}</span></div>'

                # Eight tiles now: Cost Basis + the existing seven
                k0, k1, k2, k3, k4, k5, k6, k7 = st.columns(8)

                with k0:
                    st.markdown('<div class="kpi-wrap"><div class="kpi-title">Total Cost Basis</div>'
                                f'<div class="kpi-value">{cb_sum:,.2f}</div></div>', unsafe_allow_html=True)

                with k1:
                    st.markdown('<div class="kpi-wrap"><div class="kpi-title">Total Portfolio Value</div>'
                                f'<div class="kpi-value">{mv_sum:,.2f}</div></div>', unsafe_allow_html=True)

                with k2:
                    st.markdown('<div class="kpi-wrap"><div class="kpi-title">Unrealized Gain/Loss</div></div>', unsafe_allow_html=True)
                    st.markdown(_color_text(u_sum), unsafe_allow_html=True)

                with k3:
                    st.markdown('<div class="kpi-wrap"><div class="kpi-title">Unrealized Gain/Loss %</div></div>', unsafe_allow_html=True)
                    st.markdown(_color_text(port_unreal_pct, is_pct=True), unsafe_allow_html=True)

                with k4:
                    st.markdown('<div class="kpi-wrap"><div class="kpi-title">Day\'s Value Change $</div></div>', unsafe_allow_html=True)
                    st.markdown(_color_text(day_val_sum), unsafe_allow_html=True)

                with k5:
                    st.markdown('<div class="kpi-wrap"><div class="kpi-title">Day\'s Value % Change</div></div>', unsafe_allow_html=True)
                    st.markdown(_color_text(day_pct_port, is_pct=True), unsafe_allow_html=True)

                with k6:
                    st.markdown('<div class="kpi-wrap"><div class="kpi-title">Breaks</div>'
                                f'<div class="kpi-value">{breaks}</div></div>', unsafe_allow_html=True)

                with k7:
                    st.markdown('<div class="kpi-wrap"><div class="kpi-title">Break Rate</div>'
                                f'<div class="kpi-value">{break_rate:.1f}%</div></div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # (Removed divider to eliminate extra horizontal line between KPIs and Settlement exposure)

        # Settlement exposure table (unsettled)
        _df_num = st.session_state.get("df_enriched_base")
        if isinstance(_df_num, _pd.DataFrame) and not _df_num.empty:
            # Ensure lifecycle_status exists
            if "lifecycle_status" not in _df_num.columns:
                _df_num = _derive_lifecycle_status(_df_num)
                st.session_state["df_enriched_base"] = _df_num
                st.session_state["df_enriched_display"] = build_enriched_display_frame(_df_num)

            try:
                _unsettled = _df_num[_df_num["lifecycle_status"].isin(["Pending", "Failed"])]
                if not _unsettled.empty and "counterparty" in _unsettled.columns:
                    _exposure = (
                        _unsettled.groupby("counterparty", dropna=False)["cost_basis"].sum().reset_index().sort_values("cost_basis", ascending=False)
                    )
                    _exposure["Exposure"] = _exposure["cost_basis"].apply(_fmt_money)
                    with st.expander("Settlement exposure by counterparty (unsettled)"):
                        st.dataframe(
                            _exposure[["counterparty", "Exposure"]],
                            hide_index=True,
                            use_container_width=True,
                        )
            except Exception:
                pass

        # Add bottom margin below Settlement exposure section
        st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

        # --- Final display frame prepared earlier ---
        # In-page controls: Thresholds | Day Over Day | Exceptions (relocated from sidebar)
        try:
            globals()["apply_dod"] = False
        except Exception:
            pass

        ctrl = st.container()
        with ctrl:
            c1, c2, c3 = st.columns([1, 1, 1])

            # Column 1 — Break Threshold Settings
            with c1:
                st.markdown("**Break Threshold Settings**")
                _only_breaks_flag_local = bool(st.session_state.get("only_breaks", False))
                # Absolute threshold selector (narrowed width via centered nested column)
                _a_left, _a_spacer1, _a_spacer2 = st.columns([2, 1, 1])
                with _a_left:
                    abs_choice_local = st.selectbox(
                        "Select Absolute Break Threshold:",
                        options=list(abs_options.keys()),
                        index=list(abs_options.keys()).index(_default_abs_label(float(st.session_state.get("abs_threshold", 999.0)))),
                        help="Breaks are flagged if Unrealized $ is greater than or equal to this amount.",
                        disabled=_only_breaks_flag_local,
                        key="abs_threshold_select_inpage",
                    )
                if not _only_breaks_flag_local:
                    st.session_state["abs_threshold"] = float(abs_options[abs_choice_local])

                # Percentage threshold selector (narrowed width via centered nested column)
                _p_left, _p_spacer1, _p_spacer2 = st.columns([2, 1, 1])
                with _p_left:
                    pct_choice_local = st.selectbox(
                        "Select Percentage Break Threshold:",
                        options=list(pct_options.keys()),
                        index=list(pct_options.keys()).index(_default_pct_label(float(st.session_state.get("pct_threshold", 5.0)))),
                        help="Basis points (bps). 100 bps = 1.00%. Breaks are flagged if Unrealized % is greater than or equal to this.",
                        disabled=_only_breaks_flag_local,
                        key="pct_threshold_select_inpage",
                    )
                if not _only_breaks_flag_local:
                    st.session_state["pct_threshold"] = float(pct_options[pct_choice_local])

                # Mirror tolerances exactly as before
                try:
                    st.session_state["abs_tolerance"] = float(st.session_state.get("abs_threshold", 0.0))
                    st.session_state["pct_tolerance"] = float(st.session_state.get("pct_threshold", 0.0))
                except Exception:
                    pass

                # Rule caption and active-thresholds note when disabled
                st.caption("Exception flag if: Unrealized $ ≥ abs threshold OR Unrealized % ≥ bps threshold.")
                if _only_breaks_flag_local:
                    st.caption(
                        f"Active thresholds — Absolute: {float(st.session_state.get('abs_threshold', 0)):.0f} | Percent: {float(st.session_state.get('pct_threshold', 0)):.2f}%"
                    )

            # Column 2 — Day Over Day Settings
            with c2:
                st.markdown("**Day Over Day Settings**")
                use_custom_prev_local = bool(st.session_state.get("use_custom_dod", False))
                st.session_state["use_custom_dod"] = st.toggle(
                    "Use custom day-over-day dates",
                    value=st.session_state["use_custom_dod"],
                    key="use_custom_dod_inpage",
                )

                apply_dod_local = False
                if st.session_state["use_custom_dod"]:
                    st.session_state["dod_start_date"] = st.date_input(
                        "Start Date",
                        value=st.session_state["dod_start_date"],
                        key="dod_start_date_inpage",
                    )
                    st.session_state["dod_end_date"] = st.date_input(
                        "End Date",
                        value=st.session_state["dod_end_date"],
                        key="dod_end_date_inpage",
                    )
                    apply_dod_local = st.button("Apply date range", key="apply_dod_button_inpage")

                # Trigger recompute when the toggle flips
                if use_custom_prev_local != bool(st.session_state.get("use_custom_dod", False)):
                    apply_dod_local = True

                # Set recompute flags if requested
                if apply_dod_local:
                    try:
                        globals()["apply_dod"] = True
                    except Exception:
                        pass
                    st.session_state["force_recompute_dod"] = True

            # Column 3 — Exceptions View
            with c3:
                st.markdown("**Exceptions View**")
                st.session_state["only_breaks"] = st.toggle(
                    "Show only breaks",
                    value=st.session_state["only_breaks"],
                    key="only_breaks_toggle_inpage",
                )
                st.caption("Filters the grid to positions breaching the active thresholds.")

            # Summary of current settings beneath the columns
            try:
                _abs_cur = float(st.session_state.get("abs_threshold", 0))
            except Exception:
                _abs_cur = 0.0
            try:
                _pct_cur = float(st.session_state.get("pct_threshold", 0))
            except Exception:
                _pct_cur = 0.0
            _use_custom = bool(st.session_state.get("use_custom_dod", False))
            _start = st.session_state.get("dod_start_date")
            _end = st.session_state.get("dod_end_date")
            _only = bool(st.session_state.get("only_breaks", False))
            dod_txt = (f" | DoD: {_start} → {_end}" if _use_custom and _start and _end else "")
            st.caption(
                f"Active thresholds: ${_abs_cur:,.0f} / {_pct_cur:.2f}%{dod_txt} | Only breaks: {'On' if _only else 'Off'}"
            )
        _df_disp_render = st.session_state["df_enriched_display"].copy()
        if bool(st.session_state.get("only_breaks")) and "Break" in _df_disp_render.columns:
            _df_disp_render = _df_disp_render[_df_disp_render["Break"] == "🚨"]

        # === PATCH START: Enriched Positions renderer ===
        # existing variables available here:
        #   _df_disp_render (display DataFrame), st.session_state["df_enriched_base"] (numeric)
        _df_disp = _df_disp_render

        if _df_disp is not None and not _df_disp.empty:
            available_cols = list(_df_disp.columns)

            # ensure session defaults (unchanged from your code)
            if "visible_cols" not in st.session_state:
                st.session_state["visible_cols"] = available_cols[:]
            if "pinned_cols" not in st.session_state:
                st.session_state["pinned_cols"] = ["Ticker"] if "Ticker" in available_cols else []

            st.subheader("Enriched Positions")

            # Build the two new broker-style columns using cached numeric data.
            # Guarded so nothing breaks if DoD columns are unavailable.
            _df_num_cached = st.session_state.get("df_enriched_base")
            if isinstance(_df_num_cached, _pd.DataFrame) and not _df_num_cached.empty:
                # Align numeric series to the same index/order as the display frame
                _num_aligned = _df_num_cached.loc[_df_disp.index]

                _qty = _pd.to_numeric(_num_aligned.get("quantity_num"), errors="coerce")
                _day_val = _pd.to_numeric(_num_aligned.get("Day Value Change $"), errors="coerce")
                _day_pct = _pd.to_numeric(_num_aligned.get("Day Value Change %"), errors="coerce")

                # Compute day price $ change safely
                with _pd.option_context('mode.use_inf_as_na', True):
                    valid_qty = _qty.where(_qty > 0)
                    _day_px_chg_num = _day_val.astype("float64").divide(valid_qty.astype("float64"))

                # Insert the new columns right after "Market Price" if present
                if "Market Price" in _df_disp.columns:
                    insert_at = list(_df_disp.columns).index("Market Price") + 1
                else:
                    insert_at = len(_df_disp.columns)

                if "Day's Price $ Change" not in _df_disp.columns:
                    _df_disp.insert(insert_at, "Day's Price $ Change", _day_px_chg_num.apply(_fmt_money_signed))
                    insert_at += 1
                if "Day's Price % Change" not in _df_disp.columns:
                    _df_disp.insert(insert_at, "Day's Price % Change", _day_pct.apply(_fmt_pct_signed))

                # Ensure newly added columns are included in the visible set
                try:
                    if "visible_cols" not in st.session_state or not isinstance(st.session_state.get("visible_cols"), list):
                        st.session_state["visible_cols"] = list(_df_disp.columns)
                    else:
                        for _new_col in ["Day's Price $ Change", "Day's Price % Change"]:
                            if _new_col not in st.session_state["visible_cols"]:
                                st.session_state["visible_cols"].append(_new_col)
                except Exception:
                    pass

                # Reorder to ensure "Day Value Change $" appears immediately after "Market Value"
                try:
                    if ("Market Value" in _df_disp.columns) and ("Day Value Change $" in _df_disp.columns):
                        cols_order = list(_df_disp.columns)
                        dv_pos = cols_order.index("Day Value Change $")
                        mv_pos = cols_order.index("Market Value")
                        if dv_pos != mv_pos + 1:
                            cols_order.pop(dv_pos)
                            cols_order.insert(mv_pos + 1, "Day Value Change $")
                            _df_disp = _df_disp[cols_order]
                except Exception:
                    pass

                # Build styling based on numeric values (not the formatted strings)
                def _color_series(num_s: _pd.Series):
                    out = []
                    for v in num_s:
                        if _pd.isna(v):
                            out.append("")
                        elif v > 0:
                            out.append("color: #137333; font-weight: 600;")  # green
                        elif v < 0:
                            out.append("color: #b3261e; font-weight: 600;")  # red
                        else:
                            out.append("")
                    return out

                # Reference numeric series for each styled column
                _ugl = _pd.to_numeric(_num_aligned.get("unrealized_gain_loss"), errors="coerce")
                _ugl_pct = _pd.to_numeric(_num_aligned.get("unrealized_gain_loss_pct"), errors="coerce")

                styler = _df_disp.style
                if "Day's Price $ Change" in _df_disp.columns:
                    styler = styler.apply(lambda s: _color_series(_day_px_chg_num.loc[s.index]), subset=["Day's Price $ Change"])
                if "Day's Price % Change" in _df_disp.columns:
                    styler = styler.apply(lambda s: _color_series(_day_pct.loc[s.index]), subset=["Day's Price % Change"])
                if "Day Value Change $" in _df_disp.columns:
                    styler = styler.apply(lambda s: _color_series(_day_val.loc[s.index]), subset=["Day Value Change $"])
                if "Unrealized Gain/Loss" in _df_disp.columns:
                    styler = styler.apply(lambda s: _color_series(_ugl.loc[s.index]), subset=["Unrealized Gain/Loss"])
                if "Unrealized Gain/Loss %" in _df_disp.columns:
                    styler = styler.apply(lambda s: _color_series(_ugl_pct.loc[s.index]), subset=["Unrealized Gain/Loss %"])
            else:
                # No numeric cache available; render without new columns or styling
                styler = _df_disp.style

            # apply visibility + pinning (unchanged)
            _visible = [c for c in _df_disp.columns if c in set(st.session_state["visible_cols"]) | set(st.session_state["pinned_cols"])]
            # Remove the percent DoD column from visibility if present
            _visible = [c for c in _visible if c != "Day Value Change %"]
            if not _visible:
                _visible = available_cols[:]
            _pins = [c for c in st.session_state["pinned_cols"] if c in _visible]
            col_cfg = {c: st.column_config.Column(pinned=True) for c in _pins}

            # Prepare subset of columns to display, rebuild styler on this subset
            _df_for_view = _df_disp[_visible].head(1000)
            styler = _df_for_view.style
            # Re-apply coloring if the numeric series exist
            if "_color_series" in locals():
                if "Day's Price $ Change" in _df_for_view.columns and "_day_px_chg_num" in locals():
                    styler = styler.apply(lambda s: _color_series(_day_px_chg_num.loc[s.index]), subset=["Day's Price $ Change"])
                if "Day's Price % Change" in _df_for_view.columns and "_day_pct" in locals():
                    styler = styler.apply(lambda s: _color_series(_day_pct.loc[s.index]), subset=["Day's Price % Change"])
                if "Day Value Change $" in _df_for_view.columns and "_day_val" in locals():
                    styler = styler.apply(lambda s: _color_series(_day_val.loc[s.index]), subset=["Day Value Change $"])
                if "Unrealized Gain/Loss" in _df_for_view.columns and "_ugl" in locals():
                    styler = styler.apply(lambda s: _color_series(_ugl.loc[s.index]), subset=["Unrealized Gain/Loss"])
                if "Unrealized Gain/Loss %" in _df_for_view.columns and "_ugl_pct" in locals():
                    styler = styler.apply(lambda s: _color_series(_ugl_pct.loc[s.index]), subset=["Unrealized Gain/Loss %"])

            # Render the styled table
            if bool(st.session_state.get("only_breaks")):
                st.caption(
                    f"Active thresholds — Absolute: {float(st.session_state.get('abs_threshold', 0)):.0f} | Percent: {float(st.session_state.get('pct_threshold', 0)):.2f}%"
                )
            st.dataframe(
                styler,
                use_container_width=True,
                hide_index=True,
                column_config=col_cfg if col_cfg else None,
            )
        # === PATCH END

        # Missing-values warning for Day Over Day columns (if present)
        _df_numeric_cached = st.session_state.get("df_enriched_base")
        if isinstance(_df_numeric_cached, _pd.DataFrame) and {"Day Value Change $", "Day Value Change %"}.issubset(_df_numeric_cached.columns):
            _missing = _df_numeric_cached[_df_numeric_cached[["Day Value Change $", "Day Value Change %"]].isna().any(axis=1)]
            if not _missing.empty:
                st.warning(f"{len(_missing)} rows missing day-over-day values due to unavailable prices on the chosen dates. These are set to blank.")


import pandas as pd
import numpy as np

def _bx_get_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

## removed duplicate local formatters; using helpers' shared formatters

def _bx_build_row_key(row: pd.Series, cols: list[str]) -> str:
    # Use a stable composite key across uploads to avoid collisions
    parts = []
    for c in cols:
        parts.append("") if c not in row else parts.append(str(row.get(c, "")))
    return "|".join(parts)

def _bx_init_exception_maps():
    ss = st.session_state
    if "exceptions_status_map" not in ss:
        ss["exceptions_status_map"] = {}
    if "exceptions_assignee_map" not in ss:
        ss["exceptions_assignee_map"] = {}
    if "exceptions_notes_map" not in ss:
        ss["exceptions_notes_map"] = {}
    if "exceptions_first_seen_map" not in ss:
        ss["exceptions_first_seen_map"] = {}

def _bx_reset_exception_maps():
    for k in [
        "exceptions_status_map",
        "exceptions_assignee_map",
        "exceptions_notes_map",
        "exceptions_first_seen_map",
    ]:
        st.session_state.pop(k, None)

def _bx_clean_options(values) -> list[str]:
    """Normalize option lists for multiselects by removing None/empty/placeholder sentinels."""
    cleaned = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s == "" or s.lower() in {"none", "nan", "na"}:
            continue
        cleaned.append(s)
    # Preserve order-unique then sort for a stable UI
    return sorted(list(dict.fromkeys(cleaned)))

def _bx_prepare_multi_state(key: str, options: list[str], select_all_if_empty: bool = True):
    """Ensure a multiselect's session_state value contains only valid options.
    If empty/invalid, optionally select all."""
    opts = list(options or [])
    # Drop 'None' tokens and invalids from any prior runs
    current = st.session_state.get(key, [])
    if not isinstance(current, list):
        current = []
    clean = [v for v in current if v in opts]
    if select_all_if_empty and (not clean):
        clean = opts[:]  # default to all options selected
    st.session_state[key] = clean

from typing import Any, Iterable
import math

def _bx_sanitize_options(values: Iterable[Any]) -> list[str]:
    """
    Return a cleaned, sorted, de-duplicated list of labels for widget options.
    Removes None/NaN/empty and the strings 'none','nan','na','null' (case-insensitive).
    """
    cleaned: list[str] = []
    for v in values or []:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        if s.lower() in {"none", "nan", "na", "null"}:
            continue
        cleaned.append(s)
    return sorted(list(dict.fromkeys(cleaned)))

def _bx_sync_multiselect_state(key: str, valid_options: list[str], *, default_all: bool = True) -> list[str]:
    """
    Ensure st.session_state[key] contains only values from valid_options.
    If options changed since last render, reset selections (to 'all' if requested).
    Returns the sanitized current selection list.
    """
    cache_key = f"__opts_cache__{key}"
    prev_opts = st.session_state.get(cache_key)
    changed = prev_opts != valid_options

    cur = st.session_state.get(key, [])
    if not isinstance(cur, list):
        cur = []
    cur = [v for v in cur if v in valid_options]
    if changed or (default_all and not cur):
        cur = valid_options[:]
    st.session_state[key] = cur
    st.session_state[cache_key] = valid_options[:]
    return cur

def _bx_sanitize_selection(key: str, valid: list[str], *, multi: bool = True, fallback_all: bool = True, fallback_value: str = "All"):
    """Ensure persisted widget selection in session_state contains only valid entries.
    - For multiselects: keep intersection with valid; if empty and fallback_all, set to all valid.
    - For selectbox: if current not in valid (+ optional 'All'), set to fallback_value.
    """
    ss = st.session_state
    if multi:
        cur = ss.get(key)
        if not isinstance(cur, list):
            cur = []
        cur = [v for v in cur if v in valid]
        if not cur and fallback_all:
            cur = valid[:]
        ss[key] = cur
    else:
        cur = ss.get(key)
        pool = ([fallback_value] + valid) if fallback_value else valid
        if cur not in pool:
            ss[key] = fallback_value if fallback_value in pool else (valid[0] if valid else None)

 

    # Resolve current thresholds once
    try:
        _abs_thr_cur = float(st.session_state.get("abs_threshold", st.session_state.get("abs_tolerance", 999.0)))
        _pct_thr_cur = float(st.session_state.get("pct_threshold", st.session_state.get("pct_tolerance", 5.0)))
    except Exception:
        _abs_thr_cur, _pct_thr_cur = 999.0, 5.0

    # Ensure enriched cache exists; if missing, build minimal from trades_raw.csv
    df_num_cached = st.session_state.get("df_enriched_base")
    if (not isinstance(df_num_cached, _pd.DataFrame)) or df_num_cached is None or df_num_cached.empty:
        try:
            _data_dir_seed0 = get_data_dir()
            _tr_raw0 = _data_dir_seed0 / "trades_raw.csv"
            if _tr_raw0.exists():
                _d0 = _pd.read_csv(_tr_raw0)
                def _first0(df, cols):
                    for c in cols:
                        if c in df.columns: return c
                    lo = {str(x).lower(): x for x in df.columns}
                    for c in cols:
                        if c.lower() in lo: return lo[c.lower()]
                    return None
                cs0 = _first0(_d0, ["security_id","ticker","symbol","instrument","security"]) or "security_id"
                sd0 = _first0(_d0, ["side","Side"]) or "side"
                cq0 = _first0(_d0, ["quantity","qty","Quantity"]) or "quantity"
                cp0 = _first0(_d0, ["price","Price"]) or "price"
                if (cs0 in _d0.columns) and (cq0 in _d0.columns) and (cp0 in _d0.columns):
                    w0 = _d0.copy()
                    w0[sd0] = w0.get(sd0, "").astype(str).str.upper().str.strip()
                    q0 = _pd.to_numeric(w0[cq0], errors="coerce").fillna(0)
                    p0 = _pd.to_numeric(w0[cp0], errors="coerce").fillna(0.0)
                    m0 = w0[sd0].map(lambda s: 1 if s=="BUY" else (-1 if s=="SELL" else _np.nan)).fillna(0)
                    w0["__signed_qty"] = q0 * m0
                    w0["__buy_qty"] = q0.where(w0[sd0]=="BUY").fillna(0)
                    w0["__buy_notional"] = (p0 * w0["__buy_qty"]).fillna(0.0)
                    g0 = w0.groupby(cs0, as_index=False).agg({"__signed_qty":"sum","__buy_qty":"sum","__buy_notional":"sum", cp0:"last"})
                    g0 = g0.rename(columns={"__signed_qty":"quantity_num", cp0:"market_price_num", cs0:"security_id"})
                    g0["unit_cost_num"] = _np.where(g0["__buy_qty"]>0, g0["__buy_notional"]/g0["__buy_qty"], _np.nan)
                    if g0["market_price_num"].isna().all():
                        seed0 = int(st.session_state.get("last_demo_seed", 42))
                        rng0 = _np.random.default_rng(seed0)
                        g0["market_price_num"] = _np.abs((g0["unit_cost_num"].fillna(100.0)) * (0.95 + 0.1*rng0.random(len(g0))))
                    g0["market_value"] = g0["quantity_num"] * g0["market_price_num"]
                    g0["unrealized_gain_loss"] = (g0["market_price_num"] - g0["unit_cost_num"]) * g0["quantity_num"]
                    g0 = _safe_ugl_pct(g0)
                    g0 = apply_breaks(g0, _abs_thr_cur, _pct_thr_cur)
                    st.session_state["df_enriched_base"] = g0
                    st.session_state["df_enriched_display"] = build_enriched_display_frame(g0)
                    df_num_cached = g0
        except Exception:
            pass

    # === KPI Row (additive, compact, defensive) ===
    try:
        data_dir = get_data_dir()
        raw_p = data_dir / "trades_raw.csv"
        clean_p = data_dir / "trades_clean.csv"
        records_ingested = None
        pct_stp_ready = None
        pct_passing_schema = None
        if raw_p.exists():
            try:
                _df_raw_kpi = _pd.read_csv(raw_p)
                records_ingested = len(_df_raw_kpi)
            except Exception:
                records_ingested = None
        if clean_p.exists() and (records_ingested is not None) and records_ingested > 0:
            try:
                _df_clean_kpi = _pd.read_csv(clean_p)
                pct_stp_ready = round(100.0 * len(_df_clean_kpi) / float(records_ingested), 2)
            except Exception:
                pct_stp_ready = None
    except Exception:
        records_ingested = None
        pct_stp_ready = None
        pct_passing_schema = None

    _k1, _k2, _k3 = st.columns(3)
    with _k1:
        st.metric("Records Ingested", records_ingested if records_ingested is not None else "—")
    with _k2:
        st.metric("% Passing Schema", f"{pct_passing_schema:.2f}%" if isinstance(pct_passing_schema, (int, float)) else "—")
    with _k3:
        st.metric("% STP-Ready", f"{pct_stp_ready:.2f}%" if isinstance(pct_stp_ready, (int, float)) else "—")
    try:
        _abs_thr = float(st.session_state.get("abs_threshold", st.session_state.get("abs_tolerance", 999.0)))
        _pct_thr = float(st.session_state.get("pct_threshold", st.session_state.get("pct_tolerance", 5.0)))
        st.caption(f"Simulated breaks: {int(st.session_state.get('__last_sim_breaks_count', 0))} (thresholds: ${_abs_thr:,.0f} / {_pct_thr:.1f}%)")
    except Exception:
        pass

    # === Alert Simulator (right-aligned; harmless) ===
    _c1, _c2, _c3, _c4, _c5 = st.columns([1,1,1,1,2])
    with _c5:
        # Ensure enriched cache exists so simulation has data to operate on
        try:
            if "df_enriched_base" not in st.session_state or not isinstance(st.session_state.get("df_enriched_base"), _pd.DataFrame) or st.session_state["df_enriched_base"].empty:
                _data_dir_seed = get_data_dir()
                _tr_raw_p = _data_dir_seed / "trades_raw.csv"
                if _tr_raw_p.exists():
                    _df_tr = _pd.read_csv(_tr_raw_p)
                    # Resolve columns
                    def _first_c(df, cols):
                        for c in cols:
                            if c in df.columns:
                                return c
                        lo = {str(x).lower(): x for x in df.columns}
                        for c in cols:
                            if c.lower() in lo:
                                return lo[c.lower()]
                        return None
                    _c_sym = _first_c(_df_tr, ["security_id","ticker","symbol","instrument","security"]) or "security_id"
                    _c_side = _first_c(_df_tr, ["side","Side"]) or "side"
                    _c_qty = _first_c(_df_tr, ["quantity","qty","Quantity"]) or "quantity"
                    _c_pr  = _first_c(_df_tr, ["price","Price"]) or "price"
                    if (_c_sym in _df_tr.columns) and (_c_qty in _df_tr.columns) and (_c_pr in _df_tr.columns):
                        _dfw = _df_tr.copy()
                        _dfw[_c_side] = _dfw.get(_c_side, "").astype(str).str.upper().str.strip()
                        _qty = _pd.to_numeric(_dfw[_c_qty], errors="coerce").fillna(0)
                        _pr  = _pd.to_numeric(_dfw[_c_pr], errors="coerce").fillna(0.0)
                        _mult = _dfw[_c_side].map(lambda s: 1 if s=="BUY" else (-1 if s=="SELL" else _np.nan)).fillna(0)
                        _dfw["__signed_qty"] = _qty * _mult
                        _dfw["__buy_qty"] = _qty.where(_dfw[_c_side]=="BUY").fillna(0)
                        _dfw["__buy_notional"] = (_pr * _dfw["__buy_qty"]).fillna(0.0)
                        _grp = _dfw.groupby(_c_sym, as_index=False).agg({"__signed_qty":"sum","__buy_qty":"sum","__buy_notional":"sum", _c_pr:"last"})
                        _grp = _grp.rename(columns={"__signed_qty":"quantity_num", _c_pr:"market_price_num"})
                        _grp["unit_cost_num"] = _np.where(_grp["__buy_qty"]>0, _grp["__buy_notional"]/_grp["__buy_qty"], _np.nan)
                        # Synthetic last price if missing
                        if "market_price_num" not in _grp.columns or _grp["market_price_num"].isna().all():
                            _seed = int(st.session_state.get("last_demo_seed", 42))
                            rng = _np.random.default_rng(_seed)
                            _grp["market_price_num"] = _np.abs((_grp["unit_cost_num"].fillna(100.0)) * (0.95 + 0.1*rng.random(len(_grp))))
                        _grp["market_value"] = _grp["quantity_num"] * _grp["market_price_num"]
                        _grp["unrealized_gain_loss"] = (_grp["market_price_num"] - _grp["unit_cost_num"]) * _grp["quantity_num"]
                        _den = _grp["unit_cost_num"].abs() * _grp["quantity_num"].abs()
                        _grp["unrealized_gain_loss_pct"] = _np.where((_den>0), (_grp["unrealized_gain_loss"].abs()/_den)*100.0, _np.nan)
                        # Friendly columns for display builder
                        _grp = _grp.rename(columns={_c_sym:"security_id"})
                        st.session_state["df_enriched_base"] = _grp
            # Simulate breaks on click
            if st.button("Simulate new breaks"):
                _df = st.session_state.get("df_enriched_base")
                if isinstance(_df, _pd.DataFrame) and not _df.empty:
                    base = _df.copy()
                    # choose up to 6 rows with non-zero position
                    nz = base[_pd.to_numeric(base.get("quantity_num"), errors="coerce").fillna(0)!=0]
                    sel = nz.head(6).index.tolist()
                    seed = int(st.session_state.get("last_demo_seed", 42))
                    rng = _np.random.default_rng(seed)
                    shocks = (rng.choice([-0.18, 0.18], size=len(sel)))
                    for i, sh in zip(sel, shocks):
                        try:
                            p0 = float(base.at[i, "market_price_num"])
                            p1 = max(0.01, p0 * (1.0 + sh))
                            base.at[i, "market_price_num"] = p1
                        except Exception:
                            continue
                    # Recompute derived and ensure clean pct
                    base["market_value"] = _pd.to_numeric(base["quantity_num"], errors="coerce") * _pd.to_numeric(base["market_price_num"], errors="coerce")
                    base["unrealized_gain_loss"] = (_pd.to_numeric(base["market_price_num"], errors="coerce") - _pd.to_numeric(base["unit_cost_num"], errors="coerce")) * _pd.to_numeric(base["quantity_num"], errors="coerce")
                    base = _safe_ugl_pct(base)

                    # Tighten thresholds for demo so breaks are visible
                    if float(st.session_state.get("abs_threshold", 999.0)) > 200:
                        st.session_state["abs_threshold"] = 200.0
                    if float(st.session_state.get("pct_threshold", 5.0)) > 1.0:
                        st.session_state["pct_threshold"] = 1.0
                    st.session_state["abs_tolerance"] = float(st.session_state.get("abs_threshold", 200.0))
                    st.session_state["pct_tolerance"] = float(st.session_state.get("pct_threshold", 1.0))

                    # Apply breaks and persist display
                    abs_thr_demo = float(st.session_state.get("abs_threshold", 200.0))
                    pct_thr_demo = float(st.session_state.get("pct_threshold", 1.0))
                    base = apply_breaks(base, abs_thr_demo, pct_thr_demo)
                    st.session_state["df_enriched_base"] = base
                    st.session_state["df_enriched_display"] = build_enriched_display_frame(base)
                    st.session_state["__last_sim_breaks_count"] = int((base.get("break_flag") == "🚨").sum()) if "break_flag" in base.columns else 0

                    st.toast("⚠️ Breaks simulated for demo")
                    st.rerun()
        except Exception:
            pass

    # Build from cache; if still missing, try minimal build from trades_raw.csv then proceed
    df_num = st.session_state.get("df_enriched_base")
    if (not isinstance(df_num, _pd.DataFrame)) or df_num.empty:
        try:
            _data_dir_seed2 = get_data_dir()
            _tr_raw2 = _data_dir_seed2 / "trades_raw.csv"
            if _tr_raw2.exists():
                _d = _pd.read_csv(_tr_raw2)
                # Same minimal aggregation as above
                def _first_c2(df, cols):
                    for c in cols:
                        if c in df.columns:
                            return c
                    lo = {str(x).lower(): x for x in df.columns}
                    for c in cols:
                        if c.lower() in lo:
                            return lo[c.lower()]
                    return None
                cs = _first_c2(_d, ["security_id","ticker","symbol","instrument","security"]) or "security_id"
                sd = _first_c2(_d, ["side","Side"]) or "side"
                cq = _first_c2(_d, ["quantity","qty","Quantity"]) or "quantity"
                cp = _first_c2(_d, ["price","Price"]) or "price"
                if (cs in _d.columns) and (cq in _d.columns) and (cp in _d.columns):
                    dw = _d.copy()
                    dw[sd] = dw.get(sd, "").astype(str).str.upper().str.strip()
                    qn = _pd.to_numeric(dw[cq], errors="coerce").fillna(0)
                    pn = _pd.to_numeric(dw[cp], errors="coerce").fillna(0.0)
                    mult = dw[sd].map(lambda s: 1 if s=="BUY" else (-1 if s=="SELL" else _np.nan)).fillna(0)
                    dw["__signed_qty"] = qn * mult
                    dw["__buy_qty"] = qn.where(dw[sd]=="BUY").fillna(0)
                    dw["__buy_notional"] = (pn * dw["__buy_qty"]).fillna(0.0)
                    g = dw.groupby(cs, as_index=False).agg({"__signed_qty":"sum","__buy_qty":"sum","__buy_notional":"sum", cp:"last"})
                    g = g.rename(columns={"__signed_qty":"quantity_num", cp:"market_price_num", cs:"security_id"})
                    g["unit_cost_num"] = _np.where(g["__buy_qty"]>0, g["__buy_notional"]/g["__buy_qty"], _np.nan)
                    if g["market_price_num"].isna().all():
                        seed = int(st.session_state.get("last_demo_seed", 42))
                        rng = _np.random.default_rng(seed)
                        g["market_price_num"] = _np.abs((g["unit_cost_num"].fillna(100.0)) * (0.95 + 0.1*rng.random(len(g))))
                    g["market_value"] = g["quantity_num"] * g["market_price_num"]
                    g["unrealized_gain_loss"] = (g["market_price_num"] - g["unit_cost_num"]) * g["quantity_num"]
                    den2 = g["unit_cost_num"].abs() * g["quantity_num"].abs()
                    g["unrealized_gain_loss_pct"] = _np.where(den2>0, (g["unrealized_gain_loss"].abs()/den2)*100.0, _np.nan)
                    st.session_state["df_enriched_base"] = g
                    df_num = g
        except Exception:
            pass
    if not isinstance(df_num, _pd.DataFrame) or df_num is None or df_num.empty:
        st.info("Please upload and enrich data on the Upload & Review page first.")
        return

    # Compute/refresh break flags from current thresholds
    try:
        _abs_thr = float(st.session_state.get("abs_threshold", st.session_state.get("abs_tolerance", 999.0)))
        _pct_thr = float(st.session_state.get("pct_threshold", st.session_state.get("pct_tolerance", 5.0)))
    except Exception:
        _abs_thr, _pct_thr = 999.0, 5.0
    try:
        df_num = apply_breaks(df_num, _abs_thr, _pct_thr)
        st.session_state["df_enriched_base"] = df_num
    except Exception:
        pass

    # If demo was just prepared and there are still no breaks, auto-simulate once so the page is not empty
    try:
        _has_breaks = int((df_num.get("break_flag") == "🚨").sum()) if "break_flag" in df_num.columns else 0
        if _has_breaks == 0 and bool(st.session_state.get("demo_prepared")) and not bool(st.session_state.get("__bx_auto_sim_done")):
            base = df_num.copy()
            nz = base[_pd.to_numeric(base.get("quantity_num"), errors="coerce").fillna(0)!=0]
            sel_idx = nz.head(6).index.tolist()
            seed = int(st.session_state.get("last_demo_seed", 42))
            rng = _np.random.default_rng(seed)
            shocks = (rng.choice([-0.18, 0.18], size=len(sel_idx)))
            for i, sh in zip(sel_idx, shocks):
                try:
                    p0 = float(base.at[i, "market_price_num"])
                    p1 = max(0.01, p0 * (1.0 + sh))
                    base.at[i, "market_price_num"] = p1
                except Exception:
                    continue
            base["market_value"] = _pd.to_numeric(base["quantity_num"], errors="coerce") * _pd.to_numeric(base["market_price_num"], errors="coerce")
            base["unrealized_gain_loss"] = (_pd.to_numeric(base["market_price_num"], errors="coerce") - _pd.to_numeric(base["unit_cost_num"], errors="coerce")) * _pd.to_numeric(base["quantity_num"], errors="coerce")
            base = _safe_ugl_pct(base)
            # tighten if needed
            if float(st.session_state.get("abs_threshold", 999.0)) > 500:
                st.session_state["abs_threshold"] = 500.0
            if float(st.session_state.get("pct_threshold", 5.0)) > 2.0:
                st.session_state["pct_threshold"] = 2.0
            st.session_state["abs_tolerance"] = float(st.session_state.get("abs_threshold", 500.0))
            st.session_state["pct_tolerance"] = float(st.session_state.get("pct_threshold", 2.0))
            base = apply_breaks(base, float(st.session_state["abs_threshold"]), float(st.session_state["pct_threshold"]))
            st.session_state["df_enriched_base"] = base
            st.session_state["df_enriched_display"] = build_enriched_display_frame(base)
            st.session_state["__bx_auto_sim_done"] = True
            st.session_state["__last_sim_breaks_count"] = int((base.get("break_flag") == "🚨").sum()) if "break_flag" in base.columns else 0
            st.rerun()
    except Exception:
        pass

    # Determine break column preference: break_flag → Break (fallback)
    break_col = "break_flag" if "break_flag" in df_num.columns else ("Break" if "Break" in df_num.columns else None)
    if break_col is None:
        st.success("No breaks to display with the current thresholds.")
        return

    # Base breaks set
    df_breaks_num = df_num[df_num[break_col] == "🚨"] if break_col in df_num.columns else _pd.DataFrame()
    if df_breaks_num is None or df_breaks_num.empty:
        st.success("No breaks to display with the current thresholds.")
        return

    # Resolve commonly used columns from numeric frame
    def _first(df, cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    sector_col = _first(df_breaks_num, ["Sector", "sector"])
    industry_col = _first(df_breaks_num, ["Industry", "industry"])
    ticker_col = _first(df_breaks_num, ["ticker", "Ticker"])

    # ---------------------- Filters UI (above summary and table) ----------------------
    # Sector options and selection
    if sector_col:
        sector_options = (
            _pd.Series(sorted({str(v).strip() for v in df_breaks_num[sector_col].dropna().tolist() if str(v).strip()}))
            .tolist()
        )
    else:
        sector_options = []

    # Pre-select all by default
    selected_sectors = st.multiselect(
        "Sector",
        options=sector_options,
        default=sector_options,
        key="__bx_sector_filter",
    ) if sector_options else []

    # Apply sector filter
    df_after_sector = (
        df_breaks_num[df_breaks_num[sector_col].astype(str).isin(selected_sectors)]
        if sector_col and selected_sectors else df_breaks_num
    )

    # Industry options are dependent on selected sectors
    if industry_col:
        raw_industry_vals = df_after_sector[industry_col].dropna().tolist()
        industry_options = (
            _pd.Series(sorted({str(v).strip() for v in raw_industry_vals if str(v).strip()})).tolist()
        )
    else:
        industry_options = []

    selected_industries = st.multiselect(
        "Industry",
        options=industry_options,
        default=industry_options,
        key="__bx_industry_filter",
    ) if industry_options else []

    df_after_industry = (
        df_after_sector[df_after_sector[industry_col].astype(str).isin(selected_industries)]
        if industry_col and selected_industries else df_after_sector
    )

    # Ticker search (case-insensitive substring). Empty => no filter
    ticker_query = st.text_input("Ticker search", value="", key="__bx_ticker_search")
    if ticker_col and ticker_query:
        q = str(ticker_query).strip()
        if q:
            df_after_ticker = df_after_industry[df_after_industry[ticker_col].astype(str).str.contains(q, case=False, na=False)]
        else:
            df_after_ticker = df_after_industry
    else:
        df_after_ticker = df_after_industry

    # ---------------------- Severity counts and filter ----------------------
    ugl_series_prefilter = _pd.to_numeric(df_after_ticker.get("unrealized_gain_loss"), errors="coerce").replace([_np.inf, -_np.inf], _np.nan)
    abs_ugl_pref = _np.abs(ugl_series_prefilter)

    small_mask = abs_ugl_pref < 1000
    med_mask = (abs_ugl_pref >= 1000) & (abs_ugl_pref <= 5000)
    large_mask = abs_ugl_pref > 5000

    cnt_all = int(len(df_after_ticker))
    cnt_small = int(_np.nansum(small_mask.astype(int))) if len(abs_ugl_pref) else 0
    cnt_med = int(_np.nansum(med_mask.astype(int))) if len(abs_ugl_pref) else 0
    cnt_large = int(_np.nansum(large_mask.astype(int))) if len(abs_ugl_pref) else 0

    severity_labels = [
        f"All ({cnt_all})",
        f"Small (< $1,000) ({cnt_small})",
        f"Medium ($1,000 – $5,000) ({cnt_med})",
        f"Large (> $5,000) ({cnt_large})",
    ]
    severity_choice = st.radio(
        "Severity",
        options=severity_labels,
        index=0,
        horizontal=True,
        key="__bx_severity",
    )

    # Apply severity selection
    if severity_choice.startswith("Small"):
        df_filtered = df_after_ticker[small_mask]
    elif severity_choice.startswith("Medium"):
        df_filtered = df_after_ticker[med_mask]
    elif severity_choice.startswith("Large"):
        df_filtered = df_after_ticker[large_mask]
    else:
        df_filtered = df_after_ticker

    if df_filtered is None or df_filtered.empty:
        st.success("No breaks to display with the current thresholds.")
        return

    # ---------------------- Dynamic summary (after all filters) ----------------------
    ugl_series = _pd.to_numeric(df_filtered.get("unrealized_gain_loss"), errors="coerce")
    ugl_pct_series = _pd.to_numeric(df_filtered.get("unrealized_gain_loss_pct"), errors="coerce")
    abs_ugl = _np.abs(ugl_series)
    abs_ugl_pct = _np.abs(ugl_pct_series)

    total_breaks = int(len(df_filtered))
    sum_abs_ugl = float(_np.nansum(abs_ugl)) if isinstance(abs_ugl, _pd.Series) else 0.0
    max_abs_ugl = float(_np.nanmax(abs_ugl)) if isinstance(abs_ugl, _pd.Series) and not abs_ugl.isna().all() else 0.0
    avg_abs_ugl_pct = float(_np.nanmean(abs_ugl_pct)) if isinstance(abs_ugl_pct, _pd.Series) and not abs_ugl_pct.isna().all() else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Breaks", f"{total_breaks}")
    with m2:
        st.metric("Total Abs Break ($)", _fmt_money(sum_abs_ugl))
    with m3:
        st.metric("Largest Break ($)", _fmt_money(max_abs_ugl))
    with m4:
        st.metric("Avg Break (%)", f"{avg_abs_ugl_pct:.2f}%")

    # ---------------------- Export for Ops (rollup + tip) ----------------------
    st.subheader("Export for Ops")
    if "counterparty" in df_filtered.columns:
        try:
            _roll = (
                df_filtered.groupby("counterparty", dropna=False)
                .size()
                .reset_index(name="Break Count")
                .sort_values("Break Count", ascending=False)
            )
            st.dataframe(
                _roll[["counterparty", "Break Count"]],
                hide_index=True,
                use_container_width=True,
            )
        except Exception:
            st.info("Counterparty column not available for rollup.")
    else:
        st.info("Counterparty column not available for rollup.")
    st.caption("Tip: Use the CSV with the counterparty rollup above to prioritize outreach.")

    # ---------------------- Read-only table ----------------------
    # Sort by absolute Unrealized Gain/Loss desc, then format using display builder
    sort_idx = abs_ugl.sort_values(ascending=False).index if isinstance(abs_ugl, _pd.Series) else df_filtered.index

    df_disp_all = build_enriched_display_frame(df_num)
    df_disp_filtered = df_disp_all.loc[sort_idx] if isinstance(df_disp_all, _pd.DataFrame) and not df_disp_all.empty else _pd.DataFrame()

    if df_disp_filtered is None or df_disp_filtered.empty:
        # Fallback: construct minimal display frame from numeric filtered with inline formatting
        df_disp_filtered = df_filtered.copy()
        if "quantity_num" in df_filtered.columns:
            df_disp_filtered["Quantity"] = _pd.to_numeric(df_filtered["quantity_num"], errors="coerce").apply(_fmt_qty)
        if "unit_cost_num" in df_filtered.columns:
            df_disp_filtered["Unit Cost"] = _pd.to_numeric(df_filtered["unit_cost_num"], errors="coerce").apply(_fmt_money)
        if "market_price_num" in df_filtered.columns:
            df_disp_filtered["Market Price"] = _pd.to_numeric(df_filtered["market_price_num"], errors="coerce").apply(_fmt_money)
        if "cost_basis" in df_filtered.columns:
            df_disp_filtered["Cost Basis"] = _pd.to_numeric(df_filtered["cost_basis"], errors="coerce").apply(_fmt_money)
        if "unrealized_gain_loss" in df_filtered.columns:
            df_disp_filtered["Unrealized Gain/Loss"] = _pd.to_numeric(df_filtered["unrealized_gain_loss"], errors="coerce").apply(_fmt_money_signed)
        if "unrealized_gain_loss_pct" in df_filtered.columns:
            df_disp_filtered["Unrealized Gain/Loss %"] = _pd.to_numeric(df_filtered["unrealized_gain_loss_pct"], errors="coerce").apply(_fmt_pct_signed)
        if "break_flag" in df_filtered.columns and "Break" not in df_disp_filtered.columns:
            df_disp_filtered["Break"] = df_filtered["break_flag"].astype(str)
        rename_min = {
            "ticker": "Ticker",
            "company_name": "Company Name",
            "Sector": "Sector",
            "Industry": "Industry",
            "trade_date": "Trade Date",
            "settlement_date": "Settlement Date",
            "market_date": "Market Date",
        }
        df_disp_filtered = df_disp_filtered.rename(columns={k: v for k, v in rename_min.items() if k in df_disp_filtered.columns})

    desired_cols = [
        "Ticker",
        "Company Name",
        "Sector",
        "Industry",
        "Trade Date",
        "Settlement Date",
        "Market Date",
        "Quantity",
        "Unit Cost",
        "Market Price",
        "Cost Basis",
        "Unrealized Gain/Loss",
        "Unrealized Gain/Loss %",
        "Break",
    ]
    cols_present = [c for c in desired_cols if c in df_disp_filtered.columns]
    if not cols_present:
        st.info("No displayable columns found for breaks table.")
        return

    # Assign to required integration variable name
    df_filtered_breaks = df_filtered

    # Build table to show and persistent status map
    df_to_show = df_disp_filtered[cols_present].copy()
    # Hide technical/helper columns if present
    try:
        _hide_cols = ["_key", "_tmp", "_hash", "rule_code_raw", "debug", "internal_notes"]
        to_drop = [c for c in _hide_cols if c in df_to_show.columns]
        if to_drop:
            df_to_show = df_to_show.drop(columns=to_drop)
    except Exception:
        pass

    # Ensure status map exists
    if "exceptions_status_map" not in st.session_state or not isinstance(st.session_state.get("exceptions_status_map"), dict):
        st.session_state["exceptions_status_map"] = {}

    status_map = st.session_state["exceptions_status_map"]

    # Build stable row keys helper using display values when present; fallback to numeric/raw
    def _value_or(df_disp, df_raw, idx, disp_col, raw_candidates, fmt=None):
        if disp_col in df_disp.columns:
            val = df_disp.at[idx, disp_col]
            if val is not None:
                return str(val)
        for c in raw_candidates:
            if c in df_raw.columns:
                v = df_raw.at[idx, c]
                if fmt is not None:
                    try:
                        return fmt(v)
                    except Exception:
                        pass
                return "" if v is None else str(v)
        return ""

    keys = []
    for idx in df_to_show.index:
        k_ticker = _value_or(df_to_show, df_filtered_breaks, idx, "Ticker", ["ticker", "Ticker"]) 
        k_td    = _value_or(df_to_show, df_filtered_breaks, idx, "Trade Date", ["trade_date", "Trade Date"]) 
        k_uc    = _value_or(df_to_show, df_filtered_breaks, idx, "Unit Cost", ["unit_cost_num", "unit_cost", "Unit Cost"], _fmt_money) 
        k_qty   = _value_or(df_to_show, df_filtered_breaks, idx, "Quantity", ["quantity_num", "quantity", "Quantity"], _fmt_qty)
        keys.append(f"{k_ticker}|{k_td}|{k_uc}|{k_qty}")

    # Initialize Exception Status from map or default to New
    df_to_show["Exception Status"] = [status_map.get(k, "New") for k in keys]

    # Append Exception Status to end while keeping original order
    final_cols = cols_present + (["Exception Status"] if "Exception Status" not in cols_present else [])
    df_to_show = df_to_show[final_cols]

    # Build per-column config: disable all except Exception Status; apply formats/pinning defensively
    try:
        col_cfg = {}
        # Identify identifier column to pin
        _id_cands = ["ticker", "symbol", "security_id", "security", "instrument", "Ticker", "Security ID", "Security Id", "Security"]
        _id_col = None
        for k in _id_cands:
            if k in df_to_show.columns:
                _id_col = k
                break
        # Money, percent, qty, price candidates
        money_cols = [
            "Unrealized $", "unrealized_amount", "unrealized_value", "pnl", "p_l", "amount",
            "Market Value", "Cost Basis", "Day Value Change $", "Day's Price $ Change",
        ]
        pct_cols = [
            "Unrealized %", "unrealized_pct", "pnl_pct", "p_l_pct", "percent",
            "Unrealized Gain/Loss %", "Day's Price % Change",
        ]
        qty_cols = ["quantity", "qty", "Quantity"]
        price_cols = ["price", "Price", "last_price", "market_price", "close", "Market Price", "Unit Cost"]

        present_money = [c for c in money_cols if c in df_to_show.columns]
        present_pct = [c for c in pct_cols if c in df_to_show.columns]
        present_qty = [c for c in qty_cols if c in df_to_show.columns]
        present_price = [c for c in price_cols if c in df_to_show.columns]

        for c in df_to_show.columns:
            if c == "Exception Status":
                continue
            if c == _id_col:
                col_cfg[c] = st.column_config.Column(c, pinned=True, disabled=True)
            elif c in present_money:
                col_cfg[c] = st.column_config.NumberColumn(c, format="%.2f", disabled=True)
            elif c in present_pct:
                col_cfg[c] = st.column_config.NumberColumn(c, format="%.2f%%", disabled=True)
            elif c in present_qty:
                col_cfg[c] = st.column_config.NumberColumn(c, format="%d", disabled=True)
            elif c in present_price:
                col_cfg[c] = st.column_config.NumberColumn(c, format="%.2f", disabled=True)
            else:
                col_cfg[c] = st.column_config.Column(c, disabled=True)
        # Keep Exception Status editable
        col_cfg["Exception Status"] = st.column_config.SelectboxColumn(
            "Exception Status", options=["New", "In Review", "Resolved"], required=True
        )
    except Exception:
        # Fallback to simple disabling if anything goes wrong
        col_cfg = {c: st.column_config.Column(disabled=True) for c in df_to_show.columns if c != "Exception Status"}
        col_cfg["Exception Status"] = st.column_config.SelectboxColumn(
            "Exception Status", options=["New", "In Review", "Resolved"], required=True
        )

    edited = st.data_editor(
        df_to_show,
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg,
        key="__bx_breaks_editor",
    )

    # Persist edits back to session map
    if isinstance(edited, _pd.DataFrame) and "Exception Status" in edited.columns:
        # Rebuild keys from the edited (display) values to align with how we created them
        for i in edited.index:
            k_ticker = str(edited.at[i, "Ticker"]) if "Ticker" in edited.columns else ""
            k_td     = str(edited.at[i, "Trade Date"]) if "Trade Date" in edited.columns else ""
            k_uc     = str(edited.at[i, "Unit Cost"]) if "Unit Cost" in edited.columns else ""
            k_qty    = str(edited.at[i, "Quantity"]) if "Quantity" in edited.columns else ""
            rk = f"{k_ticker}|{k_td}|{k_uc}|{k_qty}"
            status_map[rk] = str(edited.at[i, "Exception Status"]) if not _pd.isna(edited.at[i, "Exception Status"]) else "New"
        st.session_state["exceptions_status_map"] = status_map

    # CSV export of currently filtered breaks including statuses, excluding key
    # Removed optional Breaks export to avoid file sprawl

 


def page_trade_lifecycle() -> None:
    """Trade Lifecycle page.

    Read-only view that surfaces pre-settlement lifecycle using artifacts
    produced by Trade Capture & Data Quality Review. Reads from session first
    and falls back to disk; never writes or mutates artifacts.
    """
    import streamlit as st
    import pandas as _pd
    from datetime import datetime as _dt, timedelta as _td
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        ZoneInfo = None  # type: ignore

    # Session-first retrieval
    df_clean = st.session_state.get("trades_clean_df")
    df_exc = st.session_state.get("trades_exceptions_df")

    # Disk fallback (read-only)
    data_dir = get_data_dir()
    def _safe_read(p):
        try:
            if p.exists():
                df = _pd.read_csv(p)
                return df if isinstance(df, _pd.DataFrame) and not df.empty else None
        except Exception:
            return None
        return None
    if df_clean is None or (hasattr(df_clean, "empty") and df_clean.empty):
        df_clean = _safe_read(data_dir / "trades_clean.csv")
    if df_exc is None or (hasattr(df_exc, "empty") and df_exc.empty):
        df_exc = _safe_read(data_dir / "trades_exceptions.csv")

    # Empty state
    if (df_clean is None or (hasattr(df_clean, "empty") and df_clean.empty)) and (df_exc is None or (hasattr(df_exc, "empty") and df_exc.empty)):
        st.info("No lifecycle data yet. Upload and validate trades in Trade Capture & Data Quality Review.")
        return

    # Compute lifecycle fields in-memory (no persistence)
    as_of = (_dt.now(ZoneInfo("America/New_York")) if ZoneInfo else _dt.utcnow()).date()
    parts = []
    if isinstance(df_clean, _pd.DataFrame) and not df_clean.empty:
        tmp = df_clean.copy()
        tmp["stp_ready_flag"] = True
        parts.append(tmp)
    if isinstance(df_exc, _pd.DataFrame) and not df_exc.empty:
        tmp = df_exc.copy()
        tmp["stp_ready_flag"] = False
        parts.append(tmp)
    df = _pd.concat(parts, ignore_index=True) if parts else _pd.DataFrame()

    # Normalize dates
    if "trade_date" in df.columns:
        df["trade_date"] = _pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    if "settlement_date" in df.columns:
        df["settlement_date"] = _pd.to_datetime(df["settlement_date"], errors="coerce").dt.date
    df = df[df["settlement_date"].notna()] if "settlement_date" in df.columns else df.iloc[0:0]
    # Normalize security_id for reliable joins
    if "security_id" in df.columns:
        try:
            df["security_id"] = df["security_id"].astype(str).str.strip().str.upper()
        except Exception:
            pass

    # Exposure
    q = _pd.to_numeric(df.get("quantity"), errors="coerce") if "quantity" in df.columns else 0
    p = _pd.to_numeric(df.get("price"), errors="coerce") if "price" in df.columns else 0
    df["exposure_abs"] = (q * p).abs().fillna(0.0) if hasattr(q, "__mul__") else 0.0

    # Weekend/holiday flag using shared helper
    def _is_weekend_or_holiday(sd_val) -> bool:
        try:
            bad, _ = weekend_or_holiday_flag(sd_val, holiday_set)
            return bool(bad)
        except Exception:
            try:
                d = _pd.to_datetime(sd_val, errors="coerce").date()
            except Exception:
                return False
            if d is None:
                return False
            return d.weekday() >= 5
    if "settlement_date" in df.columns:
        df["is_weekend_or_holiday"] = [
            _is_weekend_or_holiday(sd) for sd in df["settlement_date"].tolist()
        ]

    # Status (pre-settlement only)
    due_today = as_of
    due_tomorrow = as_of + _td(days=1)
    def _status(sd):
        if sd == due_today:
            return "Due Today"
        if sd == due_tomorrow:
            return "Due Tomorrow"
        if sd < due_today:
            return "Past-Due"
        return "Future"
    if "settlement_date" in df.columns:
        df["status"] = [_status(sd) for sd in df["settlement_date"].tolist()]

    # --- Broker tolerance defaults (read-only, session-derived) ---
    try:
        tol_abs = float(st.session_state.get("__bcf_tol_abs", 0.01))
    except Exception:
        tol_abs = 0.01
    try:
        tol_bps = float(st.session_state.get("__bcf_tol_bps", 2.0))
    except Exception:
        tol_bps = 2.0

    # --- Broker match outcomes (graceful fallback if unavailable) ---
    affirmed = _pd.DataFrame()
    mismatches = _pd.DataFrame()
    unmatched_internal = _pd.DataFrame()
    try:
        _bm = run_broker_match(tol_abs, tol_bps)
        if isinstance(_bm, dict):
            affirmed = _bm.get("affirmed", _pd.DataFrame())
            mismatches = _bm.get("mismatches", _pd.DataFrame())
            unmatched_internal = _bm.get("unmatched_internal", _pd.DataFrame())
    except Exception:
        affirmed = _pd.DataFrame()
        mismatches = _pd.DataFrame()
        unmatched_internal = _pd.DataFrame()

    # --- Create join keys on lifecycle dataframe to mirror broker-match logic ---
    if not df.empty:
        # Normalize to mirror _norm_df in run_broker_match
        if "security_id" in df.columns:
            try:
                df["security_id"] = df["security_id"].astype(str).str.strip().str.upper()
            except Exception:
                pass
        if "side" in df.columns:
            try:
                df["side"] = df["side"].astype(str).str.strip().str.upper()
            except Exception:
                pass
        if "counterparty_legal_name" in df.columns:
            try:
                df["counterparty_legal_name"] = df["counterparty_legal_name"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            except Exception:
                pass
        if "trade_date" in df.columns:
            try:
                df["trade_date"] = _pd.to_datetime(df["trade_date"], errors="coerce").dt.date
            except Exception:
                pass
        if "quantity" in df.columns:
            try:
                df["quantity"] = _pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
            except Exception:
                pass
        if "price" in df.columns:
            try:
                df["price"] = _pd.to_numeric(df["price"], errors="coerce").astype(float)
            except Exception:
                pass
            try:
                df["price_rounded_2"] = df["price"].round(2)
            except Exception:
                df["price_rounded_2"] = _pd.NA
        else:
            df["price_rounded_2"] = _pd.NA

        # Build strict and loose internal keys used by broker matching
        def _mk_key_strict(row):
            try:
                return f"{row.get('trade_date')}|{row.get('side')}|{row.get('security_id')}|{row.get('quantity')}|{row.get('counterparty_legal_name')}|{row.get('price_rounded_2')}"
            except Exception:
                return ""
        def _mk_key_loose(row):
            try:
                return f"{row.get('trade_date')}|{row.get('side')}|{row.get('security_id')}|{row.get('quantity')}|{row.get('counterparty_legal_name')}"
            except Exception:
                return ""
        try:
            df["internal_key_strict"] = df.apply(_mk_key_strict, axis=1)
            df["internal_key_loose"] = df.apply(_mk_key_loose, axis=1)
        except Exception:
            df["internal_key_strict"] = ""
            df["internal_key_loose"] = ""

        # --- Derive broker_status with precedence: Affirmed > Mismatch > Unmatched > None ---
        affirmed_keys = set()
        if isinstance(affirmed, _pd.DataFrame) and not affirmed.empty:
            for col in ["internal_key_strict", "internal_key_loose"]:
                if col in affirmed.columns:
                    affirmed_keys.update(affirmed[col].dropna().astype(str).tolist())

        mismatch_keys = set()
        if isinstance(mismatches, _pd.DataFrame) and not mismatches.empty:
            try:
                cand_cols = [c for c in mismatches.columns if c.startswith("internal_key_")]
                for c in cand_cols:
                    mismatch_keys.update(mismatches[c].dropna().astype(str).tolist())
                # Also consider suffixed internal keys from diagnostic merges
                cand_cols_ext = [c for c in mismatches.columns if c.startswith("internal_key_") or c.endswith("_int")]
                for c in cand_cols_ext:
                    try:
                        mismatch_keys.update(mismatches[c].dropna().astype(str).tolist())
                    except Exception:
                        continue
            except Exception:
                pass

        unmatched_keys = set()
        if isinstance(unmatched_internal, _pd.DataFrame) and not unmatched_internal.empty:
            for col in ["internal_key_strict", "internal_key_loose"]:
                if col in unmatched_internal.columns:
                    unmatched_keys.update(unmatched_internal[col].dropna().astype(str).tolist())

        df["broker_status"] = _pd.Series([None] * len(df))
        try:
            is_unmatched = df["internal_key_strict"].isin(unmatched_keys) | df["internal_key_loose"].isin(unmatched_keys)
            df.loc[is_unmatched, "broker_status"] = "Unmatched"
        except Exception:
            pass
        try:
            is_mismatch = df["internal_key_strict"].isin(mismatch_keys) | df["internal_key_loose"].isin(mismatch_keys)
            df.loc[is_mismatch, "broker_status"] = "Mismatch"
        except Exception:
            pass
        try:
            is_affirmed = df["internal_key_strict"].isin(affirmed_keys) | df["internal_key_loose"].isin(affirmed_keys)
            df.loc[is_affirmed, "broker_status"] = "Affirmed"
        except Exception:
            pass

        # --- Compute unified lifecycle_state with At-Risk overlay ---
        def _derive_state(row):
            try:
                base = None
                stp_ready = bool(row.get("stp_ready_flag", False))
                bstat = row.get("broker_status")
                if not stp_ready:
                    base = "Pending – Exception"
                elif bstat == "Mismatch":
                    base = "Pending – Mismatch"
                elif bstat == "Unmatched":
                    base = "Pending – Unmatched"
                elif bstat == "Affirmed" and stp_ready:
                    base = "Ready"
                else:
                    base = "Pending – Awaiting Match"

                risky = False
                status_val = str(row.get("status", ""))
                if status_val in {"Past-Due", "Due Today", "Due Tomorrow"}:
                    risky = True
                if bool(row.get("is_weekend_or_holiday", False)):
                    risky = True
                if base != "Ready" and risky:
                    return f"At-Risk – {base}"
                return base
            except Exception:
                return row.get("lifecycle_state", None)

        try:
            df["lifecycle_state"] = df.apply(_derive_state, axis=1)
        except Exception:
            df["lifecycle_state"] = None

    # Security description lookup (read-only) and merge into working df
    sec_lookup: _pd.DataFrame | None = None
    try:
        candidates = [
            data_dir / "security_master_brk_q2_2025.csv",
            data_dir / "security_master.csv",
        ]
        for _p in candidates:
            try:
                if not _p.exists():
                    continue
                _sdf = _pd.read_csv(_p)
                if not (isinstance(_sdf, _pd.DataFrame) and not _sdf.empty):
                    continue
                _sdf.columns = [str(c).strip() for c in _sdf.columns]
                # Resolve join key from master (security_id or ticker/symbol)
                key_candidates = ["security_id", "ticker", "symbol", "instrument", "security"]
                keycol = None
                for kc in key_candidates:
                    if kc in _sdf.columns:
                        keycol = kc
                        break
                if keycol is None:
                    continue
                _sdf["__norm_sid"] = _sdf[keycol].astype(str).str.strip().str.upper()
                # Resolve description-like column
                desc_col_candidates = ["description", "company_name", "Security Description", "security_description", "name"]
                chosen_desc = None
                for c in desc_col_candidates:
                    if c in _sdf.columns:
                        chosen_desc = c
                        break
                if chosen_desc is None:
                    continue
                sec_lookup = (
                    _sdf[["__norm_sid", chosen_desc]]
                        .rename(columns={"__norm_sid": "security_id", chosen_desc: "description"})
                        .drop_duplicates(subset=["security_id"])
                        .copy()
                )
                break
            except Exception:
                continue
    except Exception:
        sec_lookup = None
    if isinstance(sec_lookup, _pd.DataFrame) and not sec_lookup.empty and "security_id" in df.columns:
        try:
            df = df.merge(sec_lookup, on="security_id", how="left")
        except Exception:
            pass

    # KPIs
    counts = df["status"].value_counts(dropna=False) if "status" in df.columns else _pd.Series(dtype=int)
    sums = (df.groupby("status")["exposure_abs"].sum() if set(["status", "exposure_abs"]).issubset(df.columns) else _pd.Series(dtype=float))
    def _count(label: str) -> int:
        try:
            return int(counts.get(label, 0))
        except Exception:
            return 0
    def _sum(label: str) -> float:
        try:
            return float(sums.get(label, 0.0))
        except Exception:
            return 0.0
    weekend_count = int(df.get("is_weekend_or_holiday", _pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if "is_weekend_or_holiday" in df.columns else 0
    weekend_sum = float(df.loc[df.get("is_weekend_or_holiday", False) == True, "exposure_abs"].sum()) if set(["is_weekend_or_holiday", "exposure_abs"]).issubset(df.columns) else 0.0
    clean_n = len(df_clean) if isinstance(df_clean, _pd.DataFrame) else 0
    exc_n = len(df_exc) if isinstance(df_exc, _pd.DataFrame) else 0
    denom = clean_n + exc_n
    stp_pct = (clean_n / denom * 100.0) if denom > 0 else 0.0

    def _fmt_count(n):
        return f"{int(n):,}"
    def _fmt_money(x):
        return f"${x:,.0f}"

    st.header("Trade Lifecycle")
    st.caption("This tab provides a pre-settlement snapshot of today’s trades, highlighting progress toward straight-through processing (STP). It surfaces the items that still require human action so you can prioritize affirmations, allocations, and exception resolution.")
    # Pre-Settlement Summary (concise narrative) — render only if meaningful data exists
    try:
        _req_cols = {"trade_date", "quantity", "price", "counterparty_legal_name", "broker_status", "lifecycle_state", "settlement_date"}
        if isinstance(df, _pd.DataFrame) and not df.empty and _req_cols.issubset(set(df.columns)):
            today_et = as_of  # already computed as ET date above
            df_today = df[df["settlement_date"] == today_et].copy()
            df_past_due = df[df.get("status").astype(str) == "Past-Due"].copy() if "status" in df.columns else _pd.DataFrame()
            if (not df_today.empty) or (not df_past_due.empty):
                # Exposure preference: use exposure_abs if present; else abs(quantity*price)
                if "exposure_abs" in df_today.columns:
                    _exp = _pd.to_numeric(df_today["exposure_abs"], errors="coerce").fillna(0.0).abs()
                else:
                    _q = _pd.to_numeric(df_today.get("quantity"), errors="coerce").fillna(0)
                    _p = _pd.to_numeric(df_today.get("price"), errors="coerce").fillna(0.0)
                    _exp = (_q * _p).abs()

                trades_due_today = int(len(df_today))
                past_due_count = int(len(df_past_due))
                ready_today = int((df_today.get("lifecycle_state").astype(str) == "Ready").sum())
                pending_today = max(trades_due_today - ready_today, 0)
                unmatched_today = int((df_today.get("broker_status").astype(str) == "Unmatched").sum())
                mismatch_today = int((df_today.get("broker_status").astype(str) == "Mismatch").sum())

                at_risk_mask = (
                    df_today.get("lifecycle_state").astype(str).str.startswith("At-Risk") |
                    df_today.get("broker_status").isin(["Unmatched", "Mismatch"])
                )
                at_risk_exposure_today = float(_exp[at_risk_mask.fillna(False)].sum())

                # Past-Due at-risk exposure (same rules, scoped to Past-Due rows)
                if not df_past_due.empty:
                    if "exposure_abs" in df_past_due.columns:
                        _exp_pd = _pd.to_numeric(df_past_due["exposure_abs"], errors="coerce").fillna(0.0).abs()
                    else:
                        _q_pd = _pd.to_numeric(df_past_due.get("quantity"), errors="coerce").fillna(0)
                        _p_pd = _pd.to_numeric(df_past_due.get("price"), errors="coerce").fillna(0.0)
                        _exp_pd = (_q_pd * _p_pd).abs()
                    at_risk_mask_pd = (
                        df_past_due.get("lifecycle_state").astype(str).str.startswith("At-Risk") |
                        df_past_due.get("broker_status").isin(["Unmatched", "Mismatch"])
                    )
                    at_risk_exposure_past_due = float(_exp_pd[at_risk_mask_pd.fillna(False)].sum())
                else:
                    at_risk_exposure_past_due = 0.0

                # Top counterparties by exposure (today)
                top_cpty_str = []
                try:
                    tmp = df_today.copy()
                    tmp["__exp"] = _exp
                    by_cpty = (
                        tmp.groupby("counterparty_legal_name", dropna=False)["__exp"].sum().sort_values(ascending=False).head(3)
                    )
                    for name, val in by_cpty.items():
                        n = str(name) if (name is not None and str(name).strip() != "") else "(Unknown Counterparty)"
                        _val_fmt = _fmt_money(float(val)).replace("$", "\\$")
                        top_cpty_str.append(f"{n} ({_val_fmt})")
                except Exception:
                    top_cpty_str = []

                # Render if there are trades settling today or past-due
                if (trades_due_today > 0) or (past_due_count > 0):
                    st.subheader("Pre-Settlement Summary")
                    if trades_due_today > 0:
                        st.write(
                            f"As of {today_et}, {trades_due_today} trades settle today: {ready_today} ready, {pending_today} pending ({unmatched_today} unmatched, {mismatch_today} mismatches)."
                        )
                    if past_due_count > 0:
                        st.write(f"Past-due trades: {past_due_count}.")
                    _amt_today = _fmt_money(at_risk_exposure_today).replace("$", "\\$")
                    _amt_past = _fmt_money(at_risk_exposure_past_due).replace("$", "\\$")
                    st.markdown(
                        f"At-risk exposure totals <span style='color:#d32f2f; font-weight:600'>{_amt_today}</span> today and <span style='color:#d32f2f; font-weight:600'>{_amt_past}</span> past-due.",
                        unsafe_allow_html=True,
                    )
                    if top_cpty_str:
                        st.write("Focus on " + ", ".join(top_cpty_str) + ".")
                    # Small divider to separate narrative from stats
                    st.markdown("---")
    except Exception:
        # Silent skip on any issues
        pass
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Due Today", _fmt_count(_count("Due Today")))
        st.caption(f"Exposure: {_fmt_money(_sum('Due Today'))}")
    with c2:
        st.metric("Due Tomorrow", _fmt_count(_count("Due Tomorrow")))
        st.caption(f"Exposure: {_fmt_money(_sum('Due Tomorrow'))}")
    with c3:
        st.metric("Past-Due", _fmt_count(_count("Past-Due")))
        st.caption(f"Exposure: {_fmt_money(_sum('Past-Due'))}")
    with c4:
        st.metric("Weekend/Holiday SD", _fmt_count(weekend_count))
        st.caption(f"Exposure: {_fmt_money(weekend_sum)}")
    with c5:
        st.metric("STP-Ready %", f"{stp_pct:.1f}%")

    # Risk badges
    badges = []
    if _count("Past-Due") > 0:
        badges.append("⚠️ Past-Due present")
    if weekend_count > 0:
        badges.append("🏖️ Weekend/Holiday SD present")
    if stp_pct < 90.0 and denom > 0:
        badges.append("🟠 STP-Ready below 90%")
    if badges:
        st.markdown(" ".join(badges))

    # Drill-down: By Counterparty
    st.subheader("By Counterparty (Exposure at Risk)")
    available_statuses = sorted(df["status"].dropna().unique().tolist()) if "status" in df.columns else []
    default_statuses = [s for s in ["Due Today", "Past-Due"] if s in available_statuses] or available_statuses
    col_cf1, col_cf2, col_cf3 = st.columns([2, 2, 1])
    with col_cf1:
        status_filter_cpy = st.multiselect("Status", options=available_statuses, default=default_statuses, key="lc_status_cpy_app")
    with col_cf2:
        cpy_query = st.text_input("Search counterparty", value="", key="lc_cpy_query_app").strip()
    with col_cf3:
        top_n_cpy = st.number_input("Top", min_value=5, max_value=50, value=10, step=5, key="lc_top_cpy_app")
    df_cpy = df.copy()
    if status_filter_cpy and "status" in df_cpy.columns:
        df_cpy = df_cpy[df_cpy["status"].isin(status_filter_cpy)]
    if cpy_query and "counterparty_legal_name" in df_cpy.columns:
        df_cpy = df_cpy[df_cpy["counterparty_legal_name"].astype(str).str.contains(cpy_query, case=False, na=False)]
    if set(["counterparty_legal_name", "exposure_abs"]).issubset(df_cpy.columns):
        by_cpy = (
            df_cpy.groupby(["counterparty_legal_name", "status"], dropna=False)["exposure_abs"].sum().reset_index().sort_values("exposure_abs", ascending=False)
        )
        # Display formatting: full dollar amounts with commas
        try:
            disp_cpy = by_cpy.copy()
            disp_cpy["exposure_abs"] = disp_cpy["exposure_abs"].fillna(0.0).map(lambda v: f"$ {v:,.0f}")
        except Exception:
            disp_cpy = by_cpy
        st.dataframe(disp_cpy.head(int(top_n_cpy)), hide_index=True, use_container_width=True)
    else:
        st.info("No counterparty data available.")

    # Drill-down: By Security
    st.subheader("By Security (Exposure at Risk)")
    col_sf1, col_sf2, col_sf3 = st.columns([2, 2, 1])
    with col_sf1:
        status_filter_sec = st.multiselect("Status", options=available_statuses, default=default_statuses, key="lc_status_sec_app")
    with col_sf2:
        sec_query = st.text_input("Search security", value="", key="lc_sec_query_app").strip()
    with col_sf3:
        top_n_sec = st.number_input("Top", min_value=5, max_value=50, value=10, step=5, key="lc_top_sec_app")
    df_sec = df.copy()
    if status_filter_sec and "status" in df_sec.columns:
        df_sec = df_sec[df_sec["status"].isin(status_filter_sec)]
    if sec_query and "security_id" in df_sec.columns:
        df_sec = df_sec[df_sec["security_id"].astype(str).str.contains(sec_query, case=False, na=False)]
    if set(["security_id", "exposure_abs"]).issubset(df_sec.columns):
        by_sec = (
            df_sec.groupby(["security_id", "status"], dropna=False)["exposure_abs"].sum().reset_index().sort_values("exposure_abs", ascending=False)
        )
        # Merge preloaded description from working df (added above)
        disp_sec = by_sec.copy()
        if "description" not in disp_sec.columns and "description" in df.columns:
            try:
                disp_sec = disp_sec.merge(df[["security_id", "description"]].drop_duplicates(), on="security_id", how="left")
            except Exception:
                pass
        # Place description next to security_id
        try:
            cols = disp_sec.columns.tolist()
            desired = ["security_id", "description", "status", "exposure_abs"]
            present = [c for c in desired if c in cols]
            others = [c for c in cols if c not in present]
            disp_sec = disp_sec[present + others] if present else disp_sec
        except Exception:
            pass
        # Display formatting: full dollar amounts with commas
        try:
            disp_sec["exposure_abs"] = disp_sec["exposure_abs"].fillna(0.0).map(lambda v: f"$ {v:,.0f}")
        except Exception:
            pass
        st.dataframe(disp_sec.head(int(top_n_sec)), hide_index=True, use_container_width=True)
    else:
        st.info("No security data available.")

    # All Rows expander
    with st.expander("All Rows", expanded=False):
        key_cols = [
            "trade_date", "settlement_date", "status", "is_weekend_or_holiday",
            "stp_ready_flag", "exposure_abs", "security_id", "description", "counterparty_legal_name",
            "broker_status", "lifecycle_state",
        ]
        show_cols = [c for c in key_cols if c in df.columns]
        st.dataframe(df[show_cols].head(5000), hide_index=True, use_container_width=True)

    # Note about exceptions
    if isinstance(df_exc, _pd.DataFrame) and not df_exc.empty:
        st.caption(f"{len(df_exc):,} trades are not STP-ready (see Trade Capture & Data Quality Review for rule-level reasons).")


def page_post_settlement() -> None:
    import streamlit as st
    st.header("Post-Settlement Reconciliation (Draft)")
    st.write(
        "This tab exists to close the trade lifecycle by verifying that internal, settled positions and cash movements tie out to external custodian records after settlement. In production, this area will compare internal EOD positions to custodian safekeeping balances, highlight quantity and market value breaks by account and security, and confirm cash movements by account and currency. It is intentionally left empty here to reserve the UX and copy for a future implementation phase. (UI skeleton — no file logic yet)"
    )


def page_data_quality_kpis() -> None:
    return


def page_help_about() -> None:
    """Back-compat wrapper: forward Help page to merged Overview & Help."""
    page_overview()

def page_validation_rules() -> None:
    """Validation Rules catalog page driven by validation_rules.yaml."""
    import pandas as _pd
    try:
        catalog = load_rule_catalog(_Path(__file__).resolve().parent / "validation_rules.yaml")

        # --- Utilities (timezone-aware, ET) ---
        def now_et() -> datetime:
            return datetime.now(ZoneInfo("America/New_York"))

        def parse_cutoff_et(hhmm: str) -> datetime:
            try:
                hh, mm = [int(x) for x in str(hhmm).split(":", 1)]
            except Exception:
                hh, mm = 21, 0
            return datetime.combine(date.today(), time(hour=hh, minute=mm), tzinfo=ZoneInfo("America/New_York"))

        def seconds_to_hms_str(total_seconds: int) -> str:
            s = max(int(total_seconds), 0)
            h = s // 3600
            m = (s % 3600) // 60
            sec = s % 60
            return f"{h:02d}:{m:02d}:{sec:02d}"

        def minutes_status(mins: int, warn_m: int, crit_m: int) -> tuple[str, str]:
            if mins < 0:
                return ("Overdue", "🔴")
            if mins <= crit_m:
                return ("Critical", "🔴")
            if mins <= warn_m:
                return ("Warning", "🟠")
            return ("On track", "🟢")

        def fmt_et_12h(hhmm: str) -> str:
            try:
                hh, mm = [int(x) for x in str(hhmm).split(":", 1)]
            except Exception:
                hh, mm = 21, 0
            dt_local = datetime.combine(date.today(), time(hh, mm), tzinfo=ZoneInfo("America/New_York"))
            # Cross-platform: avoid %-I (not on Windows). Use %I then strip leading zero.
            s = dt_local.strftime("%I:%M %p").lstrip("0").lower()
            return f"{s} ET"

        def allocations_target_str() -> str:
            return fmt_et_12h("19:00")

        # --- Read parameters and compute countdown ---
        _params = catalog.get("parameters") or {}
        cutoff_24 = str(_params.get("ctm_cutoff_time_et", "21:00"))
        try:
            warn_m = int(_params.get("ctm_warn_minutes", 120))
        except Exception:
            warn_m = 120
        try:
            crit_m = int(_params.get("ctm_critical_minutes", 30))
        except Exception:
            crit_m = 30

        cutoff_dt = parse_cutoff_et(cutoff_24)
        secs_left = int((cutoff_dt - now_et()).total_seconds())
        mins_left = math.ceil(secs_left / 60.0)
        status_text, dot = minutes_status(mins_left, warn_m, crit_m)
        cutoff_12 = fmt_et_12h(cutoff_24)
        alloc_12 = allocations_target_str()
        cutoff_dt_utc_iso = cutoff_dt.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")

        # --- Top header container ---
        with st.container():
            st.header("Validation Rules")

        # --- Single-row hero: spacer | KPIs | countdown ---
        try:
            heroLeftPad, heroCenterKPIs, heroRightBadge = st.columns([0.15, 0.55, 0.30], vertical_alignment="center")
        except TypeError:
            heroLeftPad, heroCenterKPIs, heroRightBadge = st.columns([0.15, 0.55, 0.30])
        
        # compute stats
        try:
            _rules_df = _pd.DataFrame(catalog.get("list", []))
        except Exception:
            _rules_df = _pd.DataFrame([])
        sev_counts = _rules_df["severity"].astype(str).str.lower().value_counts() if (isinstance(_rules_df, _pd.DataFrame) and "severity" in _rules_df.columns) else {}
        total_rules = int(len(_rules_df)) if isinstance(_rules_df, _pd.DataFrame) else 0
        c_crit = int(getattr(sev_counts, "get", lambda _k, _d=0: 0)("critical", 0))
        c_high = int(getattr(sev_counts, "get", lambda _k, _d=0: 0)("high", 0))
        
        # KPIs in the center hero column
        with heroCenterKPIs:
            try:
                k1, k2, k3 = st.columns([1, 1, 1], gap="small")
            except TypeError:
                k1, k2, k3 = st.columns([1, 1, 1])
            def _kpi(title: str, value: int | str):
                st.markdown(f"<div style='text-align:center; font-size:0.9rem; color:rgba(49,51,63,0.6);'>{title}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center; font-size:1.8rem; font-weight:700;'>{value}</div>", unsafe_allow_html=True)
            with k1:
                _kpi("Rules", total_rules)
            with k2:
                _kpi("Critical", c_crit)
            with k3:
                _kpi("High", c_high)

        # Countdown badge on the right hero column (position unchanged)
        with heroRightBadge:
                _html = """
          <div class="vrules-card" style="width:100%; max-width:100%; margin-left:auto; margin-right:0; border:1px solid rgba(49,51,63,0.2); border-radius:12px; padding:16px; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif; box-sizing:border-box;">
            <div style="font-size:12px; color:rgba(49,51,63,0.6); margin-bottom:6px;">Same day affirmation</div>
            <div id="sla-status" style="font-weight:600; font-size:20px; display:flex; align-items:center; gap:8px; margin-bottom:6px;">
              <span id="sla-dot" style="font-size:18px;">🟢</span><span id="sla-text">On track</span>
            </div>
            <div style="font-size:12px; color:rgba(49,51,63,0.8); margin-bottom:4px;">Time remaining</div>
            <div id="sla-timer" style="font-size:36px; font-weight:700; line-height:1; margin-bottom:6px;">--:--:--</div>
            <div id="sla-delta" style="display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2ff; color:#1f3a8a; font-size:12px; margin-bottom:10px;">
              ↑ Cutoff __CUTOFF12__
            </div>
            <div style="height:6px; background:rgba(49,51,63,0.1); border-radius:999px; overflow:hidden; margin-bottom:8px;">
              <div id="sla-bar" style="height:100%; width:0%; background:#1f77b4;"></div>
            </div>
            <div style="font-size:12px; color:rgba(49,51,63,0.6);">Target 90% affirmed by __CUTOFF12__ (allocations by ~__ALLOC12__)</div>
          </div>
          <script>
            const cutoffUtc = new Date('__CUTOFF_ISO__');
            const warnM = __WARN_M__;
            const critM = __CRIT_M__;
            function pad(n){ return n.toString().padStart(2,'0'); }
            function fmtHMS(diffSec){ const s=Math.max(diffSec,0); const h=Math.floor(s/3600); const m=Math.floor((s%3600)/60); const sec=Math.floor(s%60); return pad(h)+":"+pad(m)+":"+pad(sec); }
            function statusFor(mins){ if(mins < 0) return ['Overdue','🔴']; if(mins <= critM) return ['Critical','🔴']; if(mins <= warnM) return ['Warning','🟠']; return ['On track','🟢']; }
            function pctToWarn(mins){ const denom=Math.max(warnM,1); const pct=Math.min(Math.max(mins,0)/denom,1.0); return (pct*100).toFixed(1); }
            function tick(){ const now=new Date(); const diffSec=Math.floor((cutoffUtc - now)/1000); const mins=Math.ceil(diffSec/60); const st=statusFor(mins); document.getElementById('sla-timer').textContent=fmtHMS(diffSec); document.getElementById('sla-text').textContent=st[0]; document.getElementById('sla-dot').textContent=st[1]; document.getElementById('sla-bar').style.width=pctToWarn(mins)+'%'; }
            tick(); setInterval(tick,1000);
          </script>
          """
                _html = (_html
                         .replace("__CUTOFF12__", cutoff_12)
                         .replace("__ALLOC12__", alloc_12)
                         .replace("__CUTOFF_ISO__", cutoff_dt_utc_iso)
                         .replace("__WARN_M__", str(warn_m))
                         .replace("__CRIT_M__", str(crit_m)))
                components.html(_html, height=280, width=0, scrolling=False)
        
        # Small spacer under hero
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Brief guidance under hero row
        st.caption(
            f"Affirmation goal: 90 percent of trades affirmed by {cutoff_12} on trade date. Allocations completed by ~{alloc_12}. Thresholds from validation_rules.yaml."
        )
        rules = catalog.get("list", [])
        if not rules:
            st.info("No rules found in validation_rules.yaml.")
        else:
            df_rules = _pd.DataFrame(rules)
            if not df_rules.empty:
                try:
                    df_rules["Severity Label"] = [severity_badge_text(s) for s in df_rules.get("severity", _pd.Series([None] * len(df_rules)))]
                except Exception:
                    df_rules["Severity Label"] = ""
                # Optional: standardize message text for CTM row to 12-hour format (display only)
                try:
                    if "code" in df_rules.columns and "message" in df_rules.columns:
                        mask_ctm = df_rules["code"].astype(str).str.strip().str.lower() == "ctm_affirmation_sla"
                        df_rules.loc[mask_ctm, "message"] = df_rules.loc[mask_ctm, "message"].astype(str).str.replace("21:00", cutoff_12, regex=False).str.replace("19:00", alloc_12, regex=False)
                except Exception:
                    pass
                # Optional Live SLA column populated only for ctm_affirmation_sla
                try:
                    if "code" in df_rules.columns:
                        mask = df_rules["code"].astype(str).str.strip().str.lower() == "ctm_affirmation_sla"
                        live_text = f"{dot} {status_text} · cutoff {cutoff_12}"
                        df_rules.loc[mask, "Live SLA"] = live_text
                        df_rules.loc[~mask, "Live SLA"] = ""
                except Exception:
                    pass
                preferred = ["code", "Severity Label", "category", "owner", "message"]
                optional_common = ["check", "action", "fields_required"]
                # Place Live SLA right after Severity Label if present
                others = [c for c in df_rules.columns if c not in preferred + optional_common + ["Live SLA"]]
                ordered = [c for c in preferred if c in df_rules.columns]
                if "Live SLA" in df_rules.columns and "Severity Label" in ordered:
                    idx = ordered.index("Severity Label") + 1
                    ordered = ordered[:idx] + ["Live SLA"] + ordered[idx:]
                ordered = ordered + [c for c in optional_common if c in df_rules.columns] + others
                # Autofit height to show all rules without internal scroll
                try:
                    num_rows = len(df_rules)
                except Exception:
                    num_rows = 0
                row_height_px = 35
                header_height_px = 38
                padding_px = 12
                table_height = header_height_px + (num_rows * row_height_px) + padding_px if num_rows > 0 else None
                st.dataframe(
                    df_rules[ordered],
                    hide_index=True,
                    use_container_width=True,
                    height=table_height,
                )
        
    except Exception as _e:
        st.warning(f"Failed to load validation rules: {_e}")


# ===================== Reference Data (new page) =====================
def page_reference_data() -> None:
    """Reference Data page — read-only static masters from data/ CSVs.

    Scope: display two CSVs as-is with consistent styling; no enrichment or mutation.
    """
    import os as _os
    import pandas as _pd

    st.title("Reference Data")
    st.caption(
        "Static masters used by validation and trade enrichment. Security Master provides instrument attributes and settlement cycle configurations. Counterparty SSI Master provides broker standing settlement instructions such as DTC participant and cash accounts. Data is read-only."
    )

    @st.cache_data(ttl=None, show_spinner=False)
    def _load_static_csv(path: str) -> _pd.DataFrame:
        # Safe CSV read with UTF-8 and encoding error ignore; header inference is default
        return _pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")

    # Resolve file paths
    sec_path = _os.path.join("data", "security_master_brk_q2_2025.csv")
    ssi_path = _os.path.join("data", "counterparty_ssi_brokers.csv")

    # Optional row count KPIs
    sec_rows: int | None = None
    ssi_rows: int | None = None

    # A) Security Master
    st.subheader("Security Master")
    try:
        if not _os.path.exists(sec_path):
            st.warning(f"File not found: {sec_path}")
        else:
            df_security = _load_static_csv(sec_path)
            if df_security is None or df_security.empty:
                st.info("No rows found in security_master_brk_q2_2025.csv.")
            else:
                sec_rows = len(df_security)
                st.dataframe(df_security, use_container_width=True, hide_index=True)
    except Exception as _e:
        st.error(f"Failed to read security_master_brk_q2_2025.csv: {_e}")

    st.divider()

    # B) Counterparty SSI Master
    st.subheader("Counterparty SSI Master")
    try:
        if not _os.path.exists(ssi_path):
            st.warning(f"File not found: {ssi_path}")
        else:
            df_ssi = _load_static_csv(ssi_path)
            if df_ssi is None or df_ssi.empty:
                st.info("No rows found in counterparty_ssi_brokers.csv.")
            else:
                ssi_rows = len(df_ssi)
                st.dataframe(df_ssi, use_container_width=True, hide_index=True)
    except Exception as _e:
        st.error(f"Failed to read counterparty_ssi_brokers.csv: {_e}")

    # KPI strip (optional), only show if at least one dataset loaded with rows
    if (sec_rows is not None) or (ssi_rows is not None):
        k1, k2 = st.columns(2)
        with k1:
            if sec_rows is not None:
                st.metric(label="Security Master rows", value=int(sec_rows))
        with k2:
            if ssi_rows is not None:
                st.metric(label="Counterparty SSI rows", value=int(ssi_rows))


# ===================== Visuals & KPIs (new page) =====================
def page_visuals_kpi() -> None:
    return

def page_trade_capture_data_quality_gate() -> None:
    return

    # Paths
    DATA_DIR = st.session_state.get("DATA_DIR") or (Path(__file__).resolve().parent / "data")
    data_dir = DATA_DIR
    csv_path = data_dir / "security_master.csv"

    # Allow overwrite only when user explicitly opts in
    recreate = st.checkbox("Recreate", value=False, key="__tcqg_recreate")

    def fetch_overview(symbol: str, api_key: str) -> dict | None:
        """Call Alpha Vantage OVERVIEW endpoint for a single symbol.
        Returns parsed dict on success or None on failure.
        """
        try:
            resp = _requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "OVERVIEW",
                    "symbol": symbol,
                    "apikey": api_key,
                },
                timeout=20,
            )
            if not resp.ok:
                return None
            data = resp.json()
            # Minimal validation: must contain at least a symbol back
            return data if isinstance(data, dict) and data else None
        except Exception:
            return None

    # Button to generate the Security Master
    _btn_sec_master = st.button(
        "Generate Security Master",
        key="__tcqg_generate",
        help="Creates data/security_master.csv (equity instruments with MIC, sector, settlement cycle).",
    )
    st.caption("A clean reference list (IDs, MIC, sector) keeps pricing, validation, and matching consistent.")
    if _btn_sec_master:
        try:
            # Ensure data directory exists
            data_dir.mkdir(parents=True, exist_ok=True)

            # Idempotency guard
            if csv_path.exists() and not recreate:
                st.info("security_master.csv already exists. Check 'Recreate' to overwrite.")
            else:
                # Seed tickers and MIC mapping
                xnas_syms = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
                xnys_syms = ["JPM", "BAC", "XOM", "UNH", "T"]

                default_names = {
                    "AAPL": "Apple Inc.",
                    "MSFT": "Microsoft Corporation",
                    "AMZN": "Amazon.com, Inc.",
                    "GOOGL": "Alphabet Inc.",
                    "META": "Meta Platforms, Inc.",
                    "NVDA": "NVIDIA Corporation",
                    "TSLA": "Tesla, Inc.",
                    "JPM": "JPMorgan Chase & Co.",
                    "BAC": "Bank of America Corporation",
                    "XOM": "Exxon Mobil Corporation",
                    "UNH": "UnitedHealth Group Incorporated",
                    "T": "AT&T Inc.",
                }

                def _mic_for(sym: str) -> str:
                    return "XNAS" if sym in xnas_syms else ("XNYS" if sym in xnys_syms else "")

                base_rows = []
                for sym in xnas_syms + xnys_syms:
                    base_rows.append(
                        {
                            "security_id": sym,
                            "company_name": default_names.get(sym, ""),
                            "asset_class": "Equity",
                            "currency": "USD",
                            "country": "US",
                            "mic": _mic_for(sym),
                            "settlement_cycle": "T+1",
                            "sector": "",
                            "active_flag": True,
                        }
                    )

                # Optional enrichment via Alpha Vantage OVERVIEW
                api_key = _os.getenv("ALPHA_VANTAGE_KEY")
                if api_key:
                    for row in base_rows:
                        info = fetch_overview(row["security_id"], api_key)
                        _time.sleep(0.2)
                        if isinstance(info, dict) and info:
                            name_val = info.get("Name")
                            sector_val = info.get("Sector")
                            if isinstance(name_val, str) and name_val.strip():
                                row["company_name"] = name_val.strip()
                            if isinstance(sector_val, str) and sector_val.strip():
                                row["sector"] = sector_val.strip()

                # Build DataFrame in the exact column order
                cols = [
                    "security_id",
                    "company_name",
                    "asset_class",
                    "currency",
                    "country",
                    "mic",
                    "settlement_cycle",
                    "sector",
                    "active_flag",
                ]
                df_sec = _pd.DataFrame(base_rows, columns=cols)

                # Write CSV with UTF-8 encoding and header
                df_sec.to_csv(csv_path, index=False, encoding="utf-8")
                st.success(f"Security Master created: {csv_path}")
        except Exception as _e:
            st.error(f"Failed to generate Security Master: {_e}")

    # === Step 1B: Counterparty SSI Master (BEGIN)
    # Idempotency checkbox specific to SSI master
    ssi_recreate = st.checkbox("Recreate", value=False, key="__tcqg_ssi_recreate")

    ssi_csv_path = data_dir / "counterparty_ssi.csv"

    _btn_ssi_master = st.button(
        "Generate Counterparty SSI Master",
        key="__tcqg_generate_ssi",
        help="Creates data/counterparty_ssi.csv (DTC/IBAN/ABA, BIC, LEI).",
    )
    st.caption("Accurate SSIs (DTC + cash) prevent settlement fails.")
    if _btn_ssi_master:
        try:
            # Ensure data directory exists (no-op if present)
            data_dir.mkdir(parents=True, exist_ok=True)

            # Idempotency: do not overwrite unless explicitly requested
            if ssi_csv_path.exists() and not ssi_recreate:
                st.info("counterparty_ssi.csv already exists. Check 'Recreate' to overwrite.")
            else:
                today_str = date.today().isoformat()

                # Seed at least 8 rows across entities/locations with placeholder values
                rows_ssi = [
                    # Goldman Sachs & Co. LLC — two locations
                    {
                        "counterparty_legal_name": "Goldman Sachs & Co. LLC",
                        "lei": "LEI-GSCO-0000000001",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "GSDMUS33XXX",
                        "depository_account": "DTC-0051",
                        "cash_account": "ABA-026009593",
                        "currency": "USD",
                        "location_city": "New York",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },
                    {
                        "counterparty_legal_name": "Goldman Sachs & Co. LLC",
                        "lei": "LEI-GSCO-0000000001",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "GSDMUS33XXX",
                        "depository_account": "DTC-0151",
                        "cash_account": "ABA-021000021",
                        "currency": "USD",
                        "location_city": "Chicago",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },

                    # Morgan Stanley & Co. LLC — two locations
                    {
                        "counterparty_legal_name": "Morgan Stanley & Co. LLC",
                        "lei": "LEI-MSCI-0000000002",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "MSDMUS33XXX",
                        "depository_account": "DTC-0052",
                        "cash_account": "ABA-111000025",
                        "currency": "USD",
                        "location_city": "New York",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },
                    {
                        "counterparty_legal_name": "Morgan Stanley & Co. LLC",
                        "lei": "LEI-MSCI-0000000002",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "MSDMUS33XXX",
                        "depository_account": "DTC-0152",
                        "cash_account": "ABA-031100209",
                        "currency": "USD",
                        "location_city": "San Francisco",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },

                    # J.P. Morgan Securities LLC
                    {
                        "counterparty_legal_name": "J.P. Morgan Securities LLC",
                        "lei": "LEI-JPMS-0000000003",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "JPMXUS33XXX",
                        "depository_account": "DTC-0061",
                        "cash_account": "ABA-053000219",
                        "currency": "USD",
                        "location_city": "New York",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },

                    # BofA Securities, Inc.
                    {
                        "counterparty_legal_name": "BofA Securities, Inc.",
                        "lei": "LEI-BOFA-0000000004",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "BOFAXUS33XXX",
                        "depository_account": "DTC-0065",
                        "cash_account": "ABA-122105155",
                        "currency": "USD",
                        "location_city": "Charlotte",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },

                    # Barclays Capital Inc.
                    {
                        "counterparty_legal_name": "Barclays Capital Inc.",
                        "lei": "LEI-BARC-0000000005",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "BARXUS33XXX",
                        "depository_account": "DTC-0070",
                        "cash_account": "IBAN-FAKE123456789",
                        "currency": "USD",
                        "location_city": "New York",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },

                    # Citigroup Global Markets Inc.
                    {
                        "counterparty_legal_name": "Citigroup Global Markets Inc.",
                        "lei": "LEI-CITI-0000000006",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "CITXUS33XXX",
                        "depository_account": "DTC-0072",
                        "cash_account": "IBAN-FAKE987654321",
                        "currency": "USD",
                        "location_city": "New York",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },

                    # UBS Securities LLC
                    {
                        "counterparty_legal_name": "UBS Securities LLC",
                        "lei": "LEI-UBSS-0000000007",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "UBSXUS33XXX",
                        "depository_account": "DTC-0081",
                        "cash_account": "ABA-026009593",
                        "currency": "USD",
                        "location_city": "New York",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },

                    # Deutsche Bank Securities Inc.
                    {
                        "counterparty_legal_name": "Deutsche Bank Securities Inc.",
                        "lei": "LEI-DBSI-0000000008",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "DBSXUS33XXX",
                        "depository_account": "DTC-0085",
                        "cash_account": "ABA-021000021",
                        "currency": "USD",
                        "location_city": "New York",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },

                    # Jefferies LLC
                    {
                        "counterparty_legal_name": "Jefferies LLC",
                        "lei": "LEI-JEFF-0000000009",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "JEFXUS33XXX",
                        "depository_account": "DTC-0090",
                        "cash_account": "ABA-026013673",
                        "currency": "USD",
                        "location_city": "Boston",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },

                    # BNP Paribas Securities Corp.
                    {
                        "counterparty_legal_name": "BNP Paribas Securities Corp.",
                        "lei": "LEI-BNPP-0000000010",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "BNPXUS33XXX",
                        "depository_account": "DTC-0093",
                        "cash_account": "ABA-031100209",
                        "currency": "USD",
                        "location_city": "New York",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },

                    # Wells Fargo Securities, LLC
                    {
                        "counterparty_legal_name": "Wells Fargo Securities, LLC",
                        "lei": "LEI-WFSC-0000000011",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "WFSXUS33XXX",
                        "depository_account": "DTC-0095",
                        "cash_account": "ABA-122105155",
                        "currency": "USD",
                        "location_city": "Charlotte",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },

                    # Interactive Brokers LLC
                    {
                        "counterparty_legal_name": "Interactive Brokers LLC",
                        "lei": "LEI-IBKR-0000000012",
                        "market": "US Equities",
                        "depository": "DTC",
                        "custodian_bic": "IBKRUS33XXX",
                        "depository_account": "DTC-0101",
                        "cash_account": "IBAN-FAKE246813579",
                        "currency": "USD",
                        "location_city": "Chicago",
                        "location_country": "US",
                        "effective_date": today_str,
                        "active_flag": True,
                    },
                ]

                cols_ssi = [
                    "counterparty_legal_name",
                    "lei",
                    "market",
                    "depository",
                    "custodian_bic",
                    "depository_account",
                    "cash_account",
                    "currency",
                    "location_city",
                    "location_country",
                    "effective_date",
                    "active_flag",
                ]

                df_ssi = _pd.DataFrame(rows_ssi, columns=cols_ssi)
                df_ssi.to_csv(ssi_csv_path, index=False, encoding="utf-8")
                st.success(f"Counterparty SSI Master created: {ssi_csv_path}")
        except Exception as _e:
            st.error(f"Failed to generate Counterparty SSI Master: {_e}")
    # === Step 1B: Counterparty SSI Master (END)

    # Display expander with file details, if the CSV exists
    with st.expander("Security Master", expanded=True):
        st.write(f"Path: {csv_path.resolve()}")
        if csv_path.exists():
            try:
                df_prev = _pd.read_csv(csv_path)
                st.write(f"Total rows: {len(df_prev)}")
                st.caption("Quick preview to sanity-check the file before it drives later checks.")
                _col_cfg_sm = {c: st.column_config.Column() for c in df_prev.columns}
                st.dataframe(
                    df_prev.head(20),
                    use_container_width=True,
                    hide_index=True,
                    column_config=_col_cfg_sm,
                )
            except Exception as _e:
                st.error(f"Could not read CSV: {_e}")
        else:
            st.info("No security master yet. Click Generate Security Master above.")

    # Step 1B expander placed below the Security Master expander
    with st.expander("Counterparty SSI Master", expanded=True):
        st.write(f"Path: {ssi_csv_path.resolve()}")
        if ssi_csv_path.exists():
            try:
                df_ssi_prev = _pd.read_csv(ssi_csv_path)
                st.write(f"Total rows: {len(df_ssi_prev)}")
                st.caption("Verify counterparties and accounts exist before you create trades.")
                _col_cfg_ssi = {c: st.column_config.Column() for c in df_ssi_prev.columns}
                st.dataframe(
                    df_ssi_prev.head(20),
                    use_container_width=True,
                    hide_index=True,
                    column_config=_col_cfg_ssi,
                )
            except Exception as _e:
                st.error(f"Could not read CSV: {_e}")
        else:
            st.info("No SSI master yet. Click Generate Counterparty SSI Master above.")

    # === Step 4A: SSI Integrity Checks (BEGIN)
    st.subheader("SSI Integrity Checks")
    st.caption("Catch BIC/LEI/account format issues early to reduce downstream fails.")
    if st.button(
        "Run SSI Integrity Checks",
        key="__tcqg_run_ssi_checks",
        help="Displays BIC/LEI/DTC/currency/date exceptions; export optional CSV.",
    ):
        st.caption("Catch obvious BIC/LEI/account format issues early to reduce downstream settlement problems.")
        try:
            if not ssi_csv_path.exists():
                st.warning("counterparty_ssi.csv not found. Please run Step 1B first.")
            else:
                ssi_df = _pd.read_csv(ssi_csv_path)
                # Lightweight validators (display-only)
                import re as _re
                from datetime import date as _date

                def _fail(rule, reason, field, value):
                    return rule, reason, field, value

                def _check_bic(v: str):
                    if v is None or (isinstance(v, float) and _pd.isna(v)):
                        return _fail("bic_format_invalid", "BIC must be 8 or 11 chars (bank/country/location[/branch])", "custodian_bic", v)
                    s = str(v).strip().upper()
                    pat = r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$"
                    if not _re.fullmatch(pat, s):
                        return _fail("bic_format_invalid", "BIC must be 8 or 11 chars (bank/country/location[/branch])", "custodian_bic", v)
                    return None

                def _lei_mod97_ok(lei: str) -> bool:
                    if not isinstance(lei, str) or len(lei) != 20:
                        return False
                    s = lei.upper()
                    if not _re.fullmatch(r"^[A-Z0-9]{20}$", s):
                        return False
                    # Convert letters A=10 .. Z=35 and compute mod 97 == 1
                    digits = ""
                    for ch in s:
                        if ch.isdigit():
                            digits += ch
                        else:
                            digits += str(ord(ch) - 55)
                    try:
                        rem = 0
                        for ch in digits:
                            rem = (rem * 10 + int(ch)) % 97
                        return rem == 1
                    except Exception:
                        return False

                def _check_lei(v: str):
                    if v is None or (isinstance(v, float) and _pd.isna(v)):
                        return _fail("lei_invalid", "LEI must be 20 chars (with valid ISO 17442 check digits)", "lei", v)
                    s = str(v).strip().upper()
                    if not _lei_mod97_ok(s):
                        return _fail("lei_invalid", "LEI must be 20 chars (with valid ISO 17442 check digits)", "lei", v)
                    return None

                # IBAN validator (basic): length by country + mod 97 == 1
                _iban_lengths = {
                    "GB": 22, "DE": 22, "FR": 27, "ES": 24, "IT": 27, "NL": 18, "IE": 22, "BE": 16,
                    "LU": 20, "PT": 25, "CH": 21, "AT": 20, "PL": 28, "DK": 18, "NO": 15, "SE": 24,
                }

                def _iban_ok(val: str) -> bool:
                    s = str(val).replace(" ", "").replace("-", "").upper()
                    if s.startswith("IBAN"):
                        s = s[4:]
                    if not _re.match(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]+$", s):
                        return False
                    cc = s[:2]
                    if cc in _iban_lengths and len(s) != _iban_lengths[cc]:
                        return False
                    # move first 4 to end and convert letters
                    rearr = s[4:] + s[:4]
                    digits = ""
                    for ch in rearr:
                        if ch.isdigit():
                            digits += ch
                        else:
                            digits += str(ord(ch) - 55)
                    rem = 0
                    for ch in digits:
                        rem = (rem * 10 + int(ch)) % 97
                    return rem == 1

                def _aba_ok(val: str) -> bool:
                    s = str(val).replace(" ", "")
                    if s.upper().startswith("ABA-"):
                        s = s[4:]
                    if not _re.fullmatch(r"^[0-9]{9}$", s):
                        return False
                    digits = [int(d) for d in s]
                    checksum = (3 * (digits[0] + digits[3] + digits[6]) + 7 * (digits[1] + digits[4] + digits[7]) + (digits[2] + digits[5] + digits[8])) % 10
                    return checksum == 0

                def _check_cash_account(v: str):
                    if v is None or (isinstance(v, float) and _pd.isna(v)):
                        return _fail("iban_checksum_fail", "IBAN failed ISO 7064 MOD-97-10 check", "cash_account", v)
                    s = str(v).strip().upper()
                    if s.startswith("IBAN") or _re.match(r"^[A-Z]{2}[0-9]{2}", s):
                        if not _iban_ok(s):
                            return _fail("iban_checksum_fail", "IBAN failed ISO 7064 MOD-97-10 check", "cash_account", v)
                        return None
                    if s.startswith("ABA-") or _re.fullmatch(r"^[0-9]{9}$", s):
                        if not _aba_ok(s):
                            return _fail("aba_checksum_fail", "ABA routing number failed checksum", "cash_account", v)
                        return None
                    # If neither IBAN nor ABA-styled, treat as informational pass
                    return None

                def _check_dtc(row):
                    dep = str(row.get("depository", "")).strip().upper()
                    acc = row.get("depository_account")
                    if dep == "DTC":
                        if acc is None or (isinstance(acc, float) and _pd.isna(acc)):
                            return _fail("dtc_account_invalid", "DTC participant must be numeric (4-digit) or 'DTC-dddd' placeholder", "depository_account", acc)
                        s = str(acc).strip().upper()
                        if not (_re.fullmatch(r"^[0-9]{4}$", s) or _re.fullmatch(r"^DTC-[0-9]{4}$", s)):
                            return _fail("dtc_account_invalid", "DTC participant must be numeric (4-digit) or 'DTC-dddd' placeholder", "depository_account", acc)
                    return None

                def _check_location_currency(row):
                    mkt = str(row.get("market", "")).strip()
                    cur = str(row.get("currency", "")).strip().upper()
                    ctry = str(row.get("location_country", "")).strip().upper()
                    eff = row.get("effective_date")
                    if mkt == "US Equities":
                        if cur != "USD":
                            return _fail("currency_inconsistent", "Currency must be USD for US Equities", "currency", cur)
                        if ctry != "US":
                            return _fail("country_inconsistent", "Location country must be US for US Equities", "location_country", ctry)
                    # effective_date not in future
                    try:
                        if _pd.notna(eff):
                            d = _pd.to_datetime(str(eff), errors="coerce").date()
                            if d > _date.today():
                                return _fail("effective_date_future", "Effective date cannot be in the future", "effective_date", eff)
                    except Exception:
                        pass
                    return None

                # Evaluate rules in order per row (stop at first failure)
                exceptions = []
                ok_mask = []
                for _, r in ssi_df.iterrows():
                    # Rule order: BIC, LEI, cash, DTC, location/currency
                    err = _check_bic(r.get("custodian_bic"))
                    if err is None:
                        err = _check_lei(r.get("lei"))
                    if err is None:
                        err = _check_cash_account(r.get("cash_account"))
                    if err is None:
                        err = _check_dtc(r)
                    if err is None:
                        err = _check_location_currency(r)
                    if err is None:
                        ok_mask.append(True)
                    else:
                        ok_mask.append(False)
                        rule, reason, field, val = err
                        exceptions.append({
                            "Rule Code": rule,
                            "Exception Reason": reason,
                            "Field": field,
                            "Value": val,
                            **{c: r.get(c) for c in ssi_df.columns}
                        })

                ssi_ok = ssi_df[_pd.Series(ok_mask, index=ssi_df.index) == True].copy()
                ssi_exceptions = _pd.DataFrame(exceptions)

                total = int(len(ssi_df))
                passing = int(len(ssi_ok))
                pct_passing = (passing / total * 100.0) if total else 0.0
                by_rule = (
                    ssi_exceptions.groupby(["Rule Code", "Exception Reason"]).size().reset_index(name="Count") if not ssi_exceptions.empty else _pd.DataFrame({"Rule Code": [], "Exception Reason": [], "Count": []})
                )

                # KPIs
                k1, k2 = st.columns(2)
                with k1:
                    st.metric("SSI records", total)
                with k2:
                    st.metric("% Passing", f"{pct_passing:.2f}%")

                st.markdown("**Exceptions by rule**")
                st.dataframe(by_rule, hide_index=True, use_container_width=True)

                # Results expander
                with st.expander("SSI Integrity Results", expanded=True):
                    t1, t2 = st.tabs(["Clean", "Exceptions"])
                    with t1:
                        cols_ok = list(ssi_ok.columns)
                        if "counterparty_legal_name" in cols_ok:
                            cols_ok = ["counterparty_legal_name"] + [c for c in cols_ok if c != "counterparty_legal_name"]
                        cfg_ok = {c: st.column_config.Column() for c in cols_ok}
                        if "counterparty_legal_name" in cfg_ok:
                            cfg_ok["counterparty_legal_name"] = st.column_config.Column("Counterparty", pinned=True)
                        st.dataframe(ssi_ok[cols_ok] if cols_ok else ssi_ok, hide_index=True, use_container_width=True, column_config=cfg_ok)
                    with t2:
                        cols_ex = list(ssi_exceptions.columns)
                        if "counterparty_legal_name" in cols_ex:
                            cols_ex = ["counterparty_legal_name"] + [c for c in cols_ex if c != "counterparty_legal_name"]
                        cfg_ex = {c: st.column_config.Column() for c in cols_ex}
                        if "counterparty_legal_name" in cfg_ex:
                            cfg_ex["counterparty_legal_name"] = st.column_config.Column("Counterparty", pinned=True)
                        if "Exception Reason" in cfg_ex:
                            cfg_ex["Exception Reason"] = st.column_config.Column("Exception Reason")
                        st.dataframe(ssi_exceptions[cols_ex] if cols_ex else ssi_exceptions, hide_index=True, use_container_width=True, column_config=cfg_ex)
                        # Removed Exceptions CSV download to avoid file sprawl
        except Exception as _e:
            st.error(f"SSI Integrity Checks failed: {_e}")
    # === Step 4A: SSI Integrity Checks (END)

    # === Step 1C: Trades (Raw) (BEGIN)
    trades_recreate = st.checkbox("Recreate", value=False, key="__tcqg_trades_recreate")
    trades_csv_path = data_dir / "trades_raw.csv"

    def _is_weekday(d: date) -> bool:
        return d.weekday() < 5

    def add_business_days(d: date, n: int) -> date:
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        cur = d
        while remaining > 0:
            cur = cur + _timedelta(days=step)
            if _is_weekday(cur):
                remaining -= 1
        return cur

    if st.button(
        "Generate Trades (Raw)",
        key="__tcqg_generate_trades",
        help="Creates data/trades_raw.csv (~24 trades; mix of valid + intentionally bad rows).",
    ):
        st.caption("Creates a realistic blotter (valid + seeded issues) to exercise controls.")
        try:
            # Pre-checks: masters must exist
            sec_path = data_dir / "security_master.csv"
            ssi_path = data_dir / "counterparty_ssi.csv"
            if not sec_path.exists() or not ssi_path.exists():
                st.warning("Missing masters. Run Generate Security Master and Generate Counterparty SSI Master above, then retry.")
                return

            # API key presence (match wrapper expectations)
            if not _os.getenv("ALPHA_VANTAGE_KEY"):
                st.error("Alpha Vantage API key not found. Please set ALPHA_VANTAGE_KEY in your environment or .env and restart.")
                return

            # Idempotency guard
            if trades_csv_path.exists() and not trades_recreate:
                st.info("trades_raw.csv already exists. Check 'Recreate' to overwrite.")
                return

            # Load masters and filter
            sec = _pd.read_csv(sec_path)
            ssi = _pd.read_csv(ssi_path)
            if "active_flag" in sec.columns:
                sec = sec[sec["active_flag"] == True]
            if "asset_class" in sec.columns:
                sec = sec[sec["asset_class"].astype(str) == "Equity"]
            if "active_flag" in ssi.columns:
                ssi = ssi[ssi["active_flag"] == True]
            if sec.empty or ssi.empty:
                st.warning("Masters contain no active rows.")
                return

            # Prepare minimal frame for price prefetch via existing wrapper
            tickers_series = sec["security_id"].astype(str).str.strip().str.upper()
            unique_tickers = sorted(list(dict.fromkeys([t for t in tickers_series.tolist() if t])))
            if not unique_tickers:
                st.error("No equity tickers available for pricing.")
                return
            df_ready = _pd.DataFrame({
                "ticker": unique_tickers,
                "quantity": [1] * len(unique_tickers),
                "unit_cost": [0] * len(unique_tickers),
            })

            df_px = enrich_with_alpha_vantage(df_ready, ENTITLEMENT_MODE)
            if not isinstance(df_px, _pd.DataFrame) or df_px.empty:
                st.error("Failed to fetch prices from Alpha Vantage.")
                return

            # Build price/date maps
            px_by_ticker, date_by_ticker = {}, {}
            for _, r in df_px.iterrows():
                sym = str(r.get("__ticker_norm__", "")).upper().strip()
                px = r.get("market_price_num", r.get("market_price"))
                dt_raw = r.get("market_date")
                if not sym:
                    continue
                try:
                    px_val = float(px) if px not in (None, "") else None
                except Exception:
                    px_val = None
                if px_val is None or px_val <= 0:
                    continue
                # Normalize date to YYYY-MM-DD
                dt_str = ""
                if isinstance(dt_raw, str) and dt_raw:
                    dt_str = dt_raw[:10]
                else:
                    try:
                        dt_str = _pd.to_datetime(dt_raw).strftime("%Y-%m-%d")
                    except Exception:
                        dt_str = ""
                if not dt_str:
                    continue
                px_by_ticker[sym] = px_val
                date_by_ticker[sym] = dt_str

            priced_syms = list(px_by_ticker.keys())
            if len(priced_syms) < 6:
                st.error("Insufficient priced tickers (need at least 6). Try again later.")
                return

            # Build settlement cycle map
            sc_map = {}
            if {"security_id", "settlement_cycle"}.issubset(sec.columns):
                for _, r in sec.iterrows():
                    sc_map[str(r["security_id"]).upper()] = str(r["settlement_cycle"]).upper()

            # Inputs
            cpys = ssi["counterparty_legal_name"].astype(str).tolist()
            qty_choices = [100, 200, 500, 1000, 2500, 5000]

            import random as _random
            _random.seed(42)

            # === ADV-based diversification ===
            # Backfill/ensure daily series exist in cache for ADV computation
            try:
                _df_for_cache = _pd.DataFrame({"__ticker_norm__": priced_syms})
                _recomp = globals().get("_recompute_day_over_day_from_cache")
                if callable(_recomp):
                    _recomp(_df_for_cache)
            except Exception:
                pass

            daily_cache: dict = st.session_state.get("daily_cache", {})
            def _adv_for(sym: str) -> float:
                ts = daily_cache.get(sym)
                if isinstance(ts, dict) and ts:
                    try:
                        keys = sorted(ts.keys())
                        tail_keys = keys[-20:]
                        vols = []
                        for k in tail_keys:
                            row = ts.get(k, {})
                            v = row.get("6. volume") or row.get("volume")
                            if v not in (None, ""):
                                vols.append(float(v))
                        if vols:
                            return max(1.0, float(sum(vols) / len(vols)))
                    except Exception:
                        return 1.0
                return 1.0

            adv_by_ticker = {t: _adv_for(t) for t in priced_syms}

            # Diversified allocation plan
            TARGET_VALID = 16
            TICKER_CAP_RATIO = 0.25
            CPY_CAP = 4
            MIN_TICKERS = min(8, len(priced_syms))
            max_per_ticker = max(2, int(TARGET_VALID * TICKER_CAP_RATIO))

            # Rank tickers by ADV desc
            ranked = sorted(priced_syms, key=lambda t: adv_by_ticker.get(t, 1.0), reverse=True)

            planned_tickers: list[str] = []
            ticker_counts: dict[str, int] = {t: 0 for t in priced_syms}
            # Ensure breadth
            for t in ranked[:MIN_TICKERS]:
                planned_tickers.append(t)
                ticker_counts[t] += 1
            # Fill remaining slots using weighted sampling with caps
            while len(planned_tickers) < TARGET_VALID:
                # Allowed candidates under cap
                candidates = [t for t in priced_syms if ticker_counts.get(t, 0) < max_per_ticker]
                if not candidates:
                    candidates = priced_syms[:]
                weights = [adv_by_ticker.get(t, 1.0) for t in candidates]
                try:
                    choice = _random.choices(candidates, weights=weights, k=1)[0]
                except Exception:
                    choice = _random.choice(candidates)
                planned_tickers.append(choice)
                ticker_counts[choice] = ticker_counts.get(choice, 0) + 1

            # Counterparty allocation with cap
            counterparties = list(dict.fromkeys(cpys))
            counterparty_counts: dict[str, int] = {c: 0 for c in counterparties}
            def _assign_counterparty() -> str:
                # Prefer those under cap, choose least used
                under_cap = [c for c in counterparties if counterparty_counts.get(c, 0) < CPY_CAP]
                pool = under_cap if under_cap else counterparties
                c = min(pool, key=lambda x: counterparty_counts.get(x, 0))
                counterparty_counts[c] = counterparty_counts.get(c, 0) + 1
                return c

            # Generate valid rows per plan
            rows_valid = []
            for sym in planned_tickers:
                try:
                    px = px_by_ticker.get(sym)
                    td_str = date_by_ticker.get(sym)
                    if px is None or not td_str:
                        continue
                    td = _pd.to_datetime(td_str).date()
                    sc = sc_map.get(sym, "T+1")
                    bd_add = 2 if sc == "T+2" else 1
                    sd = add_business_days(td, bd_add)
                    rows_valid.append({
                        "trade_date": td.strftime("%Y-%m-%d"),
                        "settlement_date": sd.strftime("%Y-%m-%d"),
                        "side": _random.choice(["Buy", "Sell"]),
                        "quantity": int(_random.choice(qty_choices)),
                        "security_id": sym,
                        "price": float(px),
                        "counterparty_legal_name": _assign_counterparty(),
                        "dq_seed_issue": "",
                        "is_seed_bad": False,
                    })
                except Exception:
                    continue

            # Build up to 8 intentionally bad rows with diversity
            rows_bad = []
            issues = [
                "missing_security_id",
                "quantity_zero",
                "missing_price",
                "invalid_side",
                "unknown_counterparty",
                "bad_settlement_cycle",
                "weekend_settlement",
                "missing_counterparty",
            ]
            # Rotate through priced tickers and counterparties
            if not priced_syms:
                priced_syms = list(px_by_ticker.keys())
            if not counterparties:
                counterparties = ["Nonexistent Broker LLC"]
            for idx, code in enumerate(issues):
                sym = ranked[idx % len(ranked)] if ranked else (priced_syms[0] if priced_syms else "AAPL")
                td_str = date_by_ticker.get(sym, date.today().strftime("%Y-%m-%d"))
                td = _pd.to_datetime(td_str).date()
                cpty = counterparties[idx % len(counterparties)]
                px_val = px_by_ticker.get(sym, _pd.NA)
                if code == "missing_security_id":
                    rows_bad.append({
                        "trade_date": td.strftime("%Y-%m-%d"),
                        "settlement_date": add_business_days(td, 1).strftime("%Y-%m-%d"),
                        "side": "Buy",
                        "quantity": 100,
                        "security_id": "",
                        "price": _pd.NA,
                        "counterparty_legal_name": cpty,
                        "dq_seed_issue": code,
                        "is_seed_bad": True,
                    })
                elif code == "quantity_zero":
                    rows_bad.append({
                        "trade_date": td.strftime("%Y-%m-%d"),
                        "settlement_date": add_business_days(td, 1).strftime("%Y-%m-%d"),
                        "side": "Sell",
                        "quantity": 0,
                        "security_id": sym,
                        "price": float(px_val) if px_val is not _pd.NA else _pd.NA,
                        "counterparty_legal_name": cpty,
                        "dq_seed_issue": code,
                        "is_seed_bad": True,
                    })
                elif code == "missing_price":
                    rows_bad.append({
                        "trade_date": td.strftime("%Y-%m-%d"),
                        "settlement_date": add_business_days(td, 1).strftime("%Y-%m-%d"),
                        "side": "Buy",
                        "quantity": 500,
                        "security_id": sym,
                        "price": _pd.NA,
                        "counterparty_legal_name": cpty,
                        "dq_seed_issue": code,
                        "is_seed_bad": True,
                    })
                elif code == "invalid_side":
                    rows_bad.append({
                        "trade_date": td.strftime("%Y-%m-%d"),
                        "settlement_date": add_business_days(td, 1).strftime("%Y-%m-%d"),
                        "side": "BUY",
                        "quantity": 1000,
                        "security_id": sym,
                        "price": float(px_val) if px_val is not _pd.NA else _pd.NA,
                        "counterparty_legal_name": cpty,
                        "dq_seed_issue": code,
                        "is_seed_bad": True,
                    })
                elif code == "unknown_counterparty":
                    rows_bad.append({
                        "trade_date": td.strftime("%Y-%m-%d"),
                        "settlement_date": add_business_days(td, 1).strftime("%Y-%m-%d"),
                        "side": "Sell",
                        "quantity": 200,
                        "security_id": sym,
                        "price": float(px_val) if px_val is not _pd.NA else _pd.NA,
                        "counterparty_legal_name": "Nonexistent Broker LLC",
                        "dq_seed_issue": code,
                        "is_seed_bad": True,
                    })
                elif code == "bad_settlement_cycle":
                    rows_bad.append({
                        "trade_date": td.strftime("%Y-%m-%d"),
                        "settlement_date": add_business_days(td, 2).strftime("%Y-%m-%d"),
                        "side": "Buy",
                        "quantity": 2500,
                        "security_id": sym,
                        "price": float(px_val) if px_val is not _pd.NA else _pd.NA,
                        "counterparty_legal_name": cpty,
                        "dq_seed_issue": code,
                        "is_seed_bad": True,
                    })
                elif code == "weekend_settlement":
                    sat = td
                    while sat.weekday() != 5:
                        sat = sat + _timedelta(days=1)
                    rows_bad.append({
                        "trade_date": td.strftime("%Y-%m-%d"),
                        "settlement_date": sat.strftime("%Y-%m-%d"),
                        "side": "Sell",
                        "quantity": 5000,
                        "security_id": sym,
                        "price": float(px_val) if px_val is not _pd.NA else _pd.NA,
                        "counterparty_legal_name": cpty,
                        "dq_seed_issue": code,
                        "is_seed_bad": True,
                    })
                elif code == "missing_counterparty":
                    rows_bad.append({
                        "trade_date": td.strftime("%Y-%m-%d"),
                        "settlement_date": add_business_days(td, 1).strftime("%Y-%m-%d"),
                        "side": "Buy",
                        "quantity": 100,
                        "security_id": sym,
                        "price": float(px_val) if px_val is not _pd.NA else _pd.NA,
                        "counterparty_legal_name": "",
                        "dq_seed_issue": code,
                        "is_seed_bad": True,
                    })

            # Combine and shuffle to 24 rows total
            all_rows = rows_valid + rows_bad[:8]
            if len(all_rows) < 24:
                # repeat valid rows to reach 24 while keeping mix
                needed = 24 - len(all_rows)
                all_rows.extend(rows_valid[:needed])
            _random.shuffle(all_rows)

            # Build DataFrame with exact column order (+ seed columns at end)
            cols_trades = [
                "trade_date",
                "settlement_date",
                "side",
                "quantity",
                "security_id",
                "price",
                "counterparty_legal_name",
                "dq_seed_issue",
                "is_seed_bad",
            ]
            df_trades = _pd.DataFrame(all_rows, columns=cols_trades).head(24)
            df_trades.to_csv(trades_csv_path, index=False, encoding="utf-8")
            st.success(f"Trades (Raw) created: {trades_csv_path}")
        except Exception as _e:
            st.error(f"Failed to generate Trades (Raw): {_e}")

    # Trades (Raw) display
    with st.expander("Trades (Raw)", expanded=True):
        st.write(f"Path: {trades_csv_path.resolve()}")
        if trades_csv_path.exists():
            try:
                df_tr_prev = _pd.read_csv(trades_csv_path)
                st.write(f"Total rows: {len(df_tr_prev)}")
                show_all = st.checkbox("Show all rows", value=False, key="__tcqg_trades_show_all")

                # === Trades (Raw) display prep (BEGIN) ===
                df_disp = df_tr_prev.copy()

                # Friendly Exception Reason mapping (display-only)
                reason_map = {
                    "missing_security_id": "Missing security identifier",
                    "quantity_zero": "Quantity is zero",
                    "missing_price": "Missing market price",
                    "invalid_side": "Invalid side value",
                    "unknown_counterparty": "Counterparty not in SSI master",
                    "bad_settlement_cycle": "Settlement date inconsistent with T+1/T+2",
                    "weekend_settlement": "Settlement date falls on a weekend",
                    "missing_counterparty": "Missing counterparty name",
                    None: "",
                    "": "",
                }
                if "dq_seed_issue" in df_disp.columns:
                    df_disp["Exception Reason (Friendly)"] = df_disp["dq_seed_issue"].map(reason_map)
                else:
                    df_disp["Exception Reason (Friendly)"] = ""

                # Exception indicator (emoji)
                df_disp["Exception"] = df_disp.get("is_seed_bad", False).map(lambda x: "⚠️" if bool(x) else "")

                # Optional: friendlier Security (Ticker — Name)
                try:
                    _sec_path = data_dir / "security_master.csv"
                    _sec_master = _pd.read_csv(_sec_path) if _sec_path.exists() else _pd.DataFrame()
                    if not _sec_master.empty and {"security_id", "company_name"}.issubset(_sec_master.columns) and "security_id" in df_disp.columns:
                        ticker_to_name = dict(zip(_sec_master["security_id"].astype(str), _sec_master["company_name"].astype(str)))
                        df_disp["Security (Ticker — Name)"] = (
                            df_disp["security_id"].astype(str).str.strip()
                            + " — "
                            + df_disp["security_id"].map(ticker_to_name).fillna("")
                        ).str.replace(" — $", "", regex=True)
                except Exception:
                    pass

                # Display-only filter: only exceptions
                only_exceptions = st.checkbox("Show only exceptions", value=False, key="trades_only_ex")
                if only_exceptions and "is_seed_bad" in df_disp.columns:
                    df_disp = df_disp[df_disp["is_seed_bad"] == True]

                # Sort newest first, then by ticker (display-only)
                if "trade_date" in df_disp.columns:
                    try:
                        df_disp["_trade_date_sort"] = _pd.to_datetime(df_disp["trade_date"], errors="coerce")
                        df_disp = df_disp.sort_values(by=["_trade_date_sort", "security_id"], ascending=[False, True]).drop(columns="_trade_date_sort")
                    except Exception:
                        pass

                # Decide preview size
                df_show = df_disp if show_all else df_disp.head(20)

                # Column order for display-only view
                cols_display = [
                    "trade_date", "settlement_date",
                    "side", "quantity",
                    "security_id", "Security (Ticker — Name)",
                    "price", "counterparty_legal_name",
                    "Exception", "Exception Reason (Friendly)",
                ]
                cols_display = [c for c in cols_display if c in df_show.columns]
                # Ensure security_id (Ticker) first and pinned
                if "security_id" in cols_display:
                    cols_display = ["security_id"] + [c for c in cols_display if c != "security_id"]
                col_cfg_trades = {c: st.column_config.Column() for c in cols_display}
                # Label + formats
                if "security_id" in col_cfg_trades:
                    col_cfg_trades["security_id"] = st.column_config.Column("Ticker", pinned=True)
                if "price" in col_cfg_trades:
                    col_cfg_trades["price"] = st.column_config.NumberColumn("Price", format="%.2f")
                if "quantity" in col_cfg_trades:
                    col_cfg_trades["quantity"] = st.column_config.NumberColumn("Quantity", format="%d")

                df_show = df_show[cols_display]

                st.dataframe(
                    df_show,
                    hide_index=True,
                    column_order=cols_display,
                    use_container_width=True,
                    column_config=col_cfg_trades,
                )
                st.caption("Legend: ✅ within thresholds · 🚨 break · ⚠️ data missing/needs attention")
                # === Trades (Raw) display prep (END) ===
            except Exception as _e:
                st.error(f"Could not read CSV: {_e}")
        else:
            st.info("No trades yet. Click Generate Trades (Raw) above.")
    # === Step 1C: Trades (Raw) (END)

    # === Step 2: Validation & KPIs (BEGIN)
    st.subheader("Data Quality & KPIs")
    st.caption("Split trades into STP-ready vs. exceptions (schema, static refs, T+1 dates).")
    with st.expander("What this validation checks", expanded=False):
        st.markdown("- Schema: required fields present, quantity>0, price>0, side ∈ {Buy, Sell}\n- Static: security exists/active, counterparty exists/active, SSI has cash & depository\n- Dates: settlement aligns with T+1/T+2; no weekend settlement")
    if st.button(
        "Run Validation",
        key="__tcqg_run_validation",
        help="Validates schema/static/dates; outputs KPIs + trades_clean.csv and trades_exceptions.csv.",
    ):
        st.caption("Split trades into STP-Ready vs. Exceptions using presence/typing checks, SSI completeness, and T+1 date logic.")
        import pandas as _pd
        from datetime import datetime as _dt, date as _date, timedelta as _td

        data_dir = Path(__file__).resolve().parent / "data"
        sec_path = data_dir / "security_master.csv"
        ssi_path = data_dir / "counterparty_ssi.csv"
        trd_path = data_dir / "trades_raw.csv"

        # Safe load with warnings if missing
        missing_files = []
        if not sec_path.exists():
            missing_files.append(str(sec_path.name))
        if not ssi_path.exists():
            missing_files.append(str(ssi_path.name))
        if not trd_path.exists():
            missing_files.append(str(trd_path.name))
        if missing_files:
            st.warning(f"Missing required file(s): {', '.join(missing_files)}")
        else:
            try:
                sec = _pd.read_csv(sec_path)
                ssi = _pd.read_csv(ssi_path)
                trd = _pd.read_csv(trd_path)
            except Exception as _e:
                st.warning(f"Could not load CSVs: {_e}")
                sec = None
                ssi = None
                trd = None

            if isinstance(sec, _pd.DataFrame) and isinstance(ssi, _pd.DataFrame) and isinstance(trd, _pd.DataFrame):
                # Filter masters to active_flag == True where present
                if "active_flag" in sec.columns:
                    try:
                        sec = sec[sec["active_flag"] == True]
                    except Exception:
                        pass
                if "active_flag" in ssi.columns:
                    try:
                        ssi = ssi[ssi["active_flag"] == True]
                    except Exception:
                        pass

                # Build lookups
                def _norm_str_series(s: _pd.Series) -> _pd.Series:
                    return s.astype(str).str.strip()

                valid_secs = set(_norm_str_series(sec.get("security_id", _pd.Series(dtype=str))))
                settle_cycle_by_sec = {}
                if "security_id" in sec.columns and "settlement_cycle" in sec.columns:
                    settle_cycle_by_sec = {
                        str(r["security_id"]).strip(): str(r["settlement_cycle"]).strip()
                        for _, r in sec[["security_id", "settlement_cycle"]].dropna(subset=["security_id"]).iterrows()
                    }

                valid_cpys = set(_norm_str_series(ssi.get("counterparty_legal_name", _pd.Series(dtype=str))))

                # SSI completeness flags for US Equities: require depository_account and cash_account present
                ssi_us = ssi.copy()
                if "market" in ssi_us.columns:
                    ssi_us = ssi_us[ssi_us["market"].astype(str).str.strip() == "US Equities"]
                req_cols = ["counterparty_legal_name", "depository_account", "cash_account"]
                for c in req_cols:
                    if c not in ssi_us.columns:
                        ssi_us[c] = _pd.NA
                ssi_us["__has_req"] = ssi_us["depository_account"].notna() & ssi_us["cash_account"].notna()
                # Build completeness map: any row per CPY with required fields means complete
                ssi_complete_by_cpy = (
                    ssi_us.groupby("counterparty_legal_name")["__has_req"].max().to_dict()
                    if "counterparty_legal_name" in ssi_us.columns
                    else {}
                )

                # Helper to add business days (Mon-Fri only)
                def add_business_days(d: _date, n: int) -> _date:
                    step = 1 if n >= 0 else -1
                    days_to_add = abs(n)
                    cur = d
                    while days_to_add > 0:
                        cur = cur + _td(days=step)
                        if cur.weekday() < 5:  # Mon-Fri
                            days_to_add -= 1
                    return cur

                # Validation function
                def validate_trades(trd, sec, ssi):
                    df = trd.copy()
                    # Normalize string fields and parse dates
                    for col in ["security_id", "side", "counterparty_legal_name"]:
                        if col in df.columns:
                            df[col] = df[col].astype(str).str.strip()
                    # Parse dates
                    for col in ["trade_date", "settlement_date"]:
                        if col in df.columns:
                            df[col] = _pd.to_datetime(df[col], errors="coerce").dt.date

                    required_fields = [
                        "trade_date",
                        "settlement_date",
                        "side",
                        "quantity",
                        "security_id",
                        "price",
                        "counterparty_legal_name",
                    ]

                    def _row_rule_check(row):
                        # Returns: (rule_code, reason, passes_r1_r2, stp_ready)
                        # R1 Presence
                        for rf in required_fields:
                            if rf not in row or _pd.isna(row[rf]) or (isinstance(row[rf], str) and not str(row[rf]).strip()):
                                return (
                                    "presence_missing_field",
                                    "Missing required field",
                                    False,
                                    False,
                                )

                        # R2 Types/Domain
                        try:
                            qty = float(row["quantity"]) if row["quantity"] is not None else None
                        except Exception:
                            qty = None
                        if qty is None or qty <= 0:
                            return ("quantity_nonpositive", "Quantity must be > 0", False, False)
                        try:
                            price = float(row["price"]) if row["price"] is not None else None
                        except Exception:
                            price = None
                        if price is None or price <= 0:
                            return ("price_nonpositive", "Price must be > 0", False, False)
                        side = str(row["side"]).strip()
                        if side not in {"Buy", "Sell"}:
                            return ("invalid_side", "Side must be Buy or Sell", False, False)

                        # Past this point, R1 & R2 passed
                        # R3 Security Master
                        sec_id = str(row["security_id"]).strip()
                        if sec_id not in valid_secs:
                            return ("unknown_security", "Security not found or inactive", True, False)

                        # R4 Counterparty
                        cpy = str(row["counterparty_legal_name"]).strip()
                        if cpy not in valid_cpys:
                            return ("unknown_counterparty", "Counterparty not found or inactive", True, False)

                        # R5 SSI completeness (US Equities)
                        if not bool(ssi_complete_by_cpy.get(cpy, False)):
                            return ("incomplete_ssi", "Missing SSI depository/cash details for US Equities", True, False)

                        # R6 Settlement logic
                        cycle = settle_cycle_by_sec.get(sec_id, "").upper().replace(" ", "")
                        td = row["trade_date"]
                        sd = row["settlement_date"]
                        if isinstance(td, _date) and isinstance(sd, _date):
                            if cycle in {"T+1", "T+2"}:
                                n = 1 if cycle == "T+1" else 2
                                expected = add_business_days(td, n)
                                if sd != expected:
                                    return ("bad_settlement_cycle", "Settlement date does not match settlement cycle", True, False)
                        # If dates invalid, R1 would have caught; continue

                        # R7 Calendar: settlement must not be weekend or US holiday
                        bad_flag, reason_txt = weekend_or_holiday_flag(sd, holiday_set)
                        if bad_flag:
                            return ("weekend_or_holiday_settlement", "Settlement date falls on a weekend/holiday", True, False)

                        return (None, None, True, True)

                    results = df.apply(_row_rule_check, axis=1, result_type="expand")
                    results.columns = ["Rule Code", "Exception Reason", "_pass_r1_r2", "stp_ready"]
                    df = _pd.concat([df, results], axis=1)

                    # Attach YAML-backed metadata without changing detection logic
                    try:
                        _cat = []
                        _sev = []
                        _own = []
                        _reason = []
                        _sev_label = []
                        _codes = df.get("Rule Code", _pd.Series([None] * len(df)))
                        _reasons_raw = df.get("Exception Reason", _pd.Series([None] * len(df)))
                        for _code, _reason_raw in zip(_codes, _reasons_raw):
                            meta = rule_meta(_code)
                            cat = meta.get("category") if isinstance(meta, dict) else None
                            sev = meta.get("severity") if isinstance(meta, dict) else None
                            own = meta.get("owner") if isinstance(meta, dict) else None
                            msg = meta.get("message") if isinstance(meta, dict) else None
                            _cat.append(cat)
                            _sev.append(sev)
                            _own.append(own)
                            _reason.append(msg if (isinstance(msg, str) and len(msg.strip()) > 0) else _reason_raw)
                            _sev_label.append(severity_badge_text(sev))
                        if len(_cat) == len(df):
                            df["Category"] = _cat
                            df["Severity"] = _sev
                            df["Owner"] = _own
                            df["Exception Reason"] = _reason
                            df["Severity Label"] = _sev_label
                    except Exception:
                        pass

                    # match_key for later matching (present for all rows; relevant for STP-ready)
                    def _mk(row: _pd.Series) -> str:
                        td = row.get("trade_date")
                        td_str = td.isoformat() if isinstance(td, _date) else str(td)
                        side = str(row.get("side", "")).strip()
                        sec_id = str(row.get("security_id", "")).strip()
                        qty = row.get("quantity")
                        try:
                            qty_str = str(int(qty)) if float(qty).is_integer() else str(qty)
                        except Exception:
                            qty_str = str(qty)
                        try:
                            pr = float(row.get("price", _pd.NA))
                            price_str = f"{pr:.2f}"
                        except Exception:
                            price_str = str(row.get("price", ""))
                        cpy = str(row.get("counterparty_legal_name", "")).strip()
                        return f"{td_str}|{side}|{sec_id}|{qty_str}|{price_str}|{cpy}"

                    df["match_key"] = df.apply(_mk, axis=1)

                    # Split outputs
                    df_clean = df[df["stp_ready"] == True].copy()
                    df_exceptions = df[df["stp_ready"] == False].copy()

                    # KPIs
                    total = len(df)
                    schema_pass = int(df["_pass_r1_r2"].sum()) if "_pass_r1_r2" in df.columns else 0
                    stp_ready_count = int(df["stp_ready"].sum()) if "stp_ready" in df.columns else 0
                    pct_schema = (schema_pass / total * 100.0) if total else 0.0
                    pct_stp = (stp_ready_count / total * 100.0) if total else 0.0

                    exc_by_rule = (
                        df_exceptions.groupby(["Rule Code", "Exception Reason"]).size().reset_index(name="Count")
                        if len(df_exceptions) > 0
                        else _pd.DataFrame({"Rule Code": [], "Exception Reason": [], "Count": []})
                    )

                    kpis = {
                        "records_ingested": int(total),
                        "pct_passing_schema": float(round(pct_schema, 2)),
                        "pct_stp_ready": float(round(pct_stp, 2)),
                        "exceptions_by_rule": exc_by_rule,
                    }

                    # Drop helper column
                    if "_pass_r1_r2" in df.columns:
                        df.drop(columns=["_pass_r1_r2"], inplace=True)

                    return df_clean, df_exceptions, kpis

                # Run validation
                df_clean, df_exceptions, kpis = validate_trades(trd, sec, ssi)

                # KPI row
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Records Ingested", value=kpis.get("records_ingested", 0))
                with c2:
                    st.metric("% Passing Schema", value=f"{kpis.get('pct_passing_schema', 0.0):.2f}%")
                with c3:
                    st.metric("% STP-Ready", value=f"{kpis.get('pct_stp_ready', 0.0):.2f}%")

                tab1, tab2 = st.tabs(["STP-Ready", "Exceptions"])
                with tab1:
                    st.caption("These rows are ready for straight-through processing (book/affirm without manual touch).")
                    # Autosize + pinned Ticker first when present
                    cols_clean = list(df_clean.columns)
                    if "security_id" in cols_clean:
                        cols_clean = ["security_id"] + [c for c in cols_clean if c != "security_id"]
                    col_cfg_clean = {c: st.column_config.Column() for c in cols_clean}
                    if "security_id" in col_cfg_clean:
                        col_cfg_clean["security_id"] = st.column_config.Column("Ticker", pinned=True)
                    if "price" in col_cfg_clean:
                        col_cfg_clean["price"] = st.column_config.NumberColumn("Price", format="%.2f")
                    if "quantity" in col_cfg_clean:
                        col_cfg_clean["quantity"] = st.column_config.NumberColumn("Quantity", format="%d")

                    st.dataframe(
                        df_clean[cols_clean] if cols_clean else df_clean,
                        hide_index=True,
                        use_container_width=True,
                        column_config=col_cfg_clean,
                    )
                    if st.button("Download STP-Ready CSV", key="__tcqg_download_clean"):
                        try:
                            out_clean = data_dir / "trades_clean.csv"
                            df_clean.to_csv(out_clean, index=False, encoding="utf-8")
                            st.success(f"Saved: {out_clean}")
                        except Exception as _e:
                            st.error(f"Failed to save STP-Ready CSV: {_e}")

                with tab2:
                    st.caption("Focus remediation—'Rule Code' is the first control that failed.")
                    # Ensure required columns visible
                    base_cols = [
                        "Rule Code",
                        "Severity Label",
                        "Category",
                        "Severity",
                        "Owner",
                        "Exception Reason",
                        "trade_date",
                        "settlement_date",
                        "side",
                        "quantity",
                        "security_id",
                        "price",
                        "counterparty_legal_name",
                    ]
                    disp_cols = [c for c in base_cols if c in df_exceptions.columns]
                    if "security_id" in disp_cols:
                        disp_cols = ["security_id"] + [c for c in disp_cols if c != "security_id"]
                    col_cfg_exc = {c: st.column_config.Column() for c in disp_cols}
                    if "security_id" in col_cfg_exc:
                        col_cfg_exc["security_id"] = st.column_config.Column("Ticker", pinned=True)
                    if "price" in col_cfg_exc:
                        col_cfg_exc["price"] = st.column_config.NumberColumn("Price", format="%.2f")
                    if "quantity" in col_cfg_exc:
                        col_cfg_exc["quantity"] = st.column_config.NumberColumn("Quantity", format="%d")
                    if "Rule Code" in col_cfg_exc:
                        col_cfg_exc["Rule Code"] = st.column_config.Column("Rule Code")
                    if "Severity Label" in col_cfg_exc:
                        col_cfg_exc["Severity Label"] = st.column_config.Column("Severity")
                    if "Category" in col_cfg_exc:
                        col_cfg_exc["Category"] = st.column_config.Column("Category")
                    if "Severity" in col_cfg_exc:
                        col_cfg_exc["Severity"] = st.column_config.Column("Severity (raw)")
                    if "Owner" in col_cfg_exc:
                        col_cfg_exc["Owner"] = st.column_config.Column("Owner")
                    if "Exception Reason" in col_cfg_exc:
                        col_cfg_exc["Exception Reason"] = st.column_config.Column("Exception Reason")

                    df_exc_disp = df_exceptions[disp_cols] if disp_cols else df_exceptions
                    st.dataframe(
                        df_exc_disp,
                        hide_index=True,
                        use_container_width=True,
                        column_config=col_cfg_exc if disp_cols else None,
                    )
                    # Removed Exceptions CSV download to avoid file sprawl

                    st.markdown("**Exceptions by Rule**")
                    st.dataframe(
                        kpis.get("exceptions_by_rule"),
                        hide_index=True,
                        use_container_width=True,
                    )

                with st.expander("Rule Dictionary (from YAML)", expanded=False):
                    try:
                        cat = load_rule_catalog(_Path(__file__).resolve().parent / "validation_rules.yaml")
                        rules = cat.get("list", [])
                        if not rules:
                            st.info("No rules found in validation_rules.yaml.")
                        else:
                            # Order preferred columns then any extra fields
                            import pandas as _pd
                            df_rules = _pd.DataFrame(rules)
                            preferred = ["code", "category", "severity", "owner", "message"]
                            extras = [c for c in df_rules.columns if c not in preferred]
                            ordered = [c for c in preferred if c in df_rules.columns] + extras
                            st.dataframe(df_rules[ordered], hide_index=True, use_container_width=True)
                    except Exception as _e:
                        st.warning(f"Failed to render rules from YAML: {_e}")
    # === Step 2: Validation & KPIs (END)

    # Read-only: Last market data update caption (derived from trades_clean.csv if present)
    try:
        data_dir = get_data_dir()
        clean_p = data_dir / "trades_clean.csv"
        last_str = "—"
        if clean_p.exists():
            try:
                _df_clean_ts = _pd.read_csv(clean_p)
                candidates = [
                    "market_timestamp",
                    "price_timestamp",
                    "last_updated",
                    "asof",
                    "as_of",
                    "market_date",
                    "market_time",
                    "timestamp",
                ]
                lower_map = {str(c).lower(): c for c in _df_clean_ts.columns}
                chosen = None
                for k in candidates:
                    if k in lower_map:
                        chosen = lower_map[k]
                        break
                if chosen:
                    ts = _pd.to_datetime(_df_clean_ts[chosen], errors="coerce")
                    mx = ts.max()
                    if _pd.notna(mx):
                        last_str = _pd.to_datetime(mx).strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
        st.caption(f"Last market data update: {last_str}")
    except Exception:
        # Silent fail to keep page unchanged on any error
        pass

    # === Step 5A: Post-Settlement Reconciliation (Stub) (BEGIN)
    st.subheader("Post-Settlement Reconciliation (Stub)")
    st.caption("Illustrate how positions and cash would reconcile to custodian statements, context for the end-to-end workflow. Demo purposes.")
    st.caption("Step 5A: Generate mock custodian statements (positions & cash).")
    st.caption("Standards mirrored: MT535/MT536 (positions/transactions), camt.053 (cash statement).")

    data_dir = Path(__file__).resolve().parent / "data"
    cust_pos_path = data_dir / "custodian_positions.csv"
    cust_cash_path = data_dir / "custodian_cash.csv"

    col_pos = [
        "as_of_date", "safekeeping_account", "security_id", "settlement_status", "settled_qty", "price_settle", "currency",
    ]
    col_cash = [
        "value_date", "account_id", "currency", "amount", "reference", "related_security_id",
    ]

    pos_recreate = st.checkbox("Recreate", value=False, key="__tcqg_cust_pos_recreate")
    if st.button(
        "Generate Custodian Positions (Mock)",
        key="__tcqg_generate_cust_pos",
        help="Creates data/custodian_positions.csv (MT535-like).",
    ):
        try:
            if cust_pos_path.exists() and not pos_recreate:
                st.info("custodian_positions.csv already exists. Check 'Recreate' to overwrite.")
            else:
                import pandas as _pd
                # Prefer trades_clean to build settled positions
                clean_path = data_dir / "trades_clean.csv"
                rows = []
                if clean_path.exists():
                    dfc = _pd.read_csv(clean_path)
                    if not dfc.empty and {"security_id", "quantity", "price", "trade_date", "settlement_date", "side"}.issubset(dfc.columns):
                        dfc["security_id"] = dfc["security_id"].astype(str).str.upper().str.strip()
                        dfc["quantity"] = _pd.to_numeric(dfc["quantity"], errors="coerce").fillna(0).astype(int)
                        dfc["price"] = _pd.to_numeric(dfc["price"], errors="coerce").astype(float)
                        dfc["side"] = dfc["side"].astype(str).str.upper().str.strip()
                        # Settled position impact: Buy = +qty, Sell = -qty
                        dfc["signed_qty"] = dfc.apply(lambda r: int(r["quantity"]) * (1 if r["side"] == "BUY" else -1), axis=1)
                        grp = dfc.groupby("security_id", as_index=False).agg({"signed_qty": "sum", "price": "last", "settlement_date": "max"})
                        for _, r in grp.iterrows():
                            rows.append({
                                "as_of_date": str(r.get("settlement_date", ""))[:10],
                                "safekeeping_account": "CUST-0001",
                                "security_id": str(r.get("security_id", "")).strip(),
                                "settlement_status": "SETTLED",
                                "settled_qty": int(r.get("signed_qty", 0)),
                                "price_settle": float(r.get("price", 0.0)),
                                "currency": "USD",
                            })
                if not rows:
                    # Fallback placeholders using security master if present
                    sec_path = data_dir / "security_master.csv"
                    tickers = []
                    if sec_path.exists():
                        try:
                            _sec = _pd.read_csv(sec_path)
                            tickers = _sec.get("security_id", _pd.Series(dtype=str)).astype(str).str.upper().head(8).tolist()
                        except Exception:
                            tickers = []
                    if not tickers:
                        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM"]
                    today_str = _pd.Timestamp.today().strftime("%Y-%m-%d")
                    for t in tickers:
                        rows.append({
                            "as_of_date": today_str,
                            "safekeeping_account": "CUST-0001",
                            "security_id": t,
                            "settlement_status": "SETTLED",
                            "settled_qty": 100,
                            "price_settle": 100.00,
                            "currency": "USD",
                        })
                _pd.DataFrame(rows, columns=col_pos).to_csv(cust_pos_path, index=False, encoding="utf-8")
                st.success(f"Custodian positions created: {cust_pos_path}")
        except Exception as _e:
            st.error(f"Failed to generate custodian positions: {_e}")

    cash_recreate = st.checkbox("Recreate", value=False, key="__tcqg_cust_cash_recreate")
    if st.button(
        "Generate Custodian Cash (Mock)",
        key="__tcqg_generate_cust_cash",
        help="Creates data/custodian_cash.csv (camt.053-like).",
    ):
        try:
            if cust_cash_path.exists() and not cash_recreate:
                st.info("custodian_cash.csv already exists. Check 'Recreate' to overwrite.")
            else:
                import pandas as _pd
                rows = []
                clean_path = data_dir / "trades_clean.csv"
                if clean_path.exists():
                    dfc = _pd.read_csv(clean_path)
                    if not dfc.empty and {"security_id", "quantity", "price", "settlement_date", "side"}.issubset(dfc.columns):
                        dfc["security_id"] = dfc["security_id"].astype(str).str.upper().str.strip()
                        dfc["quantity"] = _pd.to_numeric(dfc["quantity"], errors="coerce").fillna(0).astype(int)
                        dfc["price"] = _pd.to_numeric(dfc["price"], errors="coerce").astype(float)
                        dfc["side"] = dfc["side"].astype(str).str.upper().str.strip()
                        for _, r in dfc.iterrows():
                            amt = float(r["quantity"]) * float(r["price"]) * (1 if r["side"] == "SELL" else -1)
                            rows.append({
                                "value_date": str(r.get("settlement_date", ""))[:10],
                                "account_id": "CASH-0001",
                                "currency": "USD",
                                "amount": float(amt),
                                "reference": f"DvP-{str(r.get('side',''))}-{str(r.get('security_id',''))}",
                                "related_security_id": str(r.get("security_id", "")).strip(),
                            })
                if not rows:
                    today_str = _pd.Timestamp.today().strftime("%Y-%m-%d")
                    rows = [
                        {"value_date": today_str, "account_id": "CASH-0001", "currency": "USD", "amount": 10000.00, "reference": "Opening credit", "related_security_id": ""},
                        {"value_date": today_str, "account_id": "CASH-0001", "currency": "USD", "amount": -5000.00, "reference": "DvP-BUY-AAPL", "related_security_id": "AAPL"},
                        {"value_date": today_str, "account_id": "CASH-0001", "currency": "USD", "amount": 2500.00, "reference": "DvP-SELL-MSFT", "related_security_id": "MSFT"},
                    ]
                _pd.DataFrame(rows, columns=col_cash).to_csv(cust_cash_path, index=False, encoding="utf-8")
                st.success(f"Custodian cash created: {cust_cash_path}")
        except Exception as _e:
            st.error(f"Failed to generate custodian cash: {_e}")

    with st.expander("Custodian Statements", expanded=True):
        # Positions tab
        tpos, tcash = st.tabs(["Positions (MT535-like)", "Cash (camt.053-like)"])
        with tpos:
            st.write(f"Path: {cust_pos_path.resolve()}")
            if cust_pos_path.exists():
                try:
                    dfp = _pd.read_csv(cust_pos_path)
                    st.write(f"Total rows: {len(dfp)}")
                    show_all_pos = st.checkbox("Show all rows", value=False, key="__tcqg_cust_pos_show_all")
                    dfp_show = dfp if show_all_pos else dfp.head(20)
                    cols_p = list(dfp_show.columns)
                    if "security_id" in cols_p:
                        cols_p = ["security_id"] + [c for c in cols_p if c != "security_id"]
                    cfg_p = {c: st.column_config.Column() for c in cols_p}
                    if "security_id" in cfg_p:
                        cfg_p["security_id"] = st.column_config.Column("Ticker", pinned=True)
                    if "price_settle" in cfg_p:
                        cfg_p["price_settle"] = st.column_config.NumberColumn("Price (Settle)", format="%.2f")
                    if "settled_qty" in cfg_p:
                        cfg_p["settled_qty"] = st.column_config.NumberColumn("Settled Qty", format="%d")
                    st.dataframe(dfp_show[cols_p] if cols_p else dfp_show, hide_index=True, use_container_width=True, column_config=cfg_p)
                except Exception as _e:
                    st.error(f"Could not read custodian_positions.csv: {_e}")
            else:
                st.info("No custodian positions yet. Click Generate Custodian Positions (Mock) above.")
        with tcash:
            st.write(f"Path: {cust_cash_path.resolve()}")
            if cust_cash_path.exists():
                try:
                    dfc = _pd.read_csv(cust_cash_path)
                    st.write(f"Total rows: {len(dfc)}")
                    show_all_cash = st.checkbox("Show all rows", value=False, key="__tcqg_cust_cash_show_all")
                    dfc_show = dfc if show_all_cash else dfc.head(20)
                    cols_c = list(dfc_show.columns)
                    if "related_security_id" in cols_c:
                        cols_c = ["related_security_id"] + [c for c in cols_c if c != "related_security_id"]
                    cfg_c = {c: st.column_config.Column() for c in cols_c}
                    if "related_security_id" in cfg_c:
                        cfg_c["related_security_id"] = st.column_config.Column("Ticker", pinned=True)
                    if "amount" in cfg_c:
                        cfg_c["amount"] = st.column_config.NumberColumn("Amount", format="%.2f")
                    st.dataframe(dfc_show[cols_c] if cols_c else dfc_show, hide_index=True, use_container_width=True, column_config=cfg_c)
                except Exception as _e:
                    st.error(f"Could not read custodian_cash.csv: {_e}")
            else:
                st.info("No custodian cash yet. Click Generate Custodian Cash (Mock) above.")
    # === Step 5A: Post-Settlement Reconciliation (Stub) (END)

    # === Step 3A: Broker Affirmation & Match — Scaffold (BEGIN)
    st.subheader("Broker Affirmation & Match")
    st.caption("Step 3A: Generate a mock broker confirm feed to set up auto-affirmation.")
    recreate_confirms = st.checkbox("Recreate", value=False, key="__tcqg_bcf_recreate")
    if st.button(
        "Generate Broker Confirms (Mock)",
        key="__tcqg_generate_bcf_mock",
        help="Creates data/broker_confirms.csv from eligible trades (venue MIC from Security Master).",
    ):
        st.caption("Produce a broker confirm feed so you can match/affirm internal trades—key for meeting T+1 timelines. Demo purposes.")
        try:
            data_dir = Path(__file__).resolve().parent / "data"
            trd_path = data_dir / "trades_raw.csv"
            ssi_path = data_dir / "counterparty_ssi.csv"
            out_path = data_dir / "broker_confirms.csv"

            # Idempotency guard
            if out_path.exists() and not recreate_confirms:
                st.info("broker_confirms.csv already exists. Check 'Recreate' to overwrite.")
                return

            # Pre-checks
            if not trd_path.exists() or not ssi_path.exists():
                st.warning("Missing masters/trades. Please run Steps 1B and 1C first.")
                return

            import pandas as _pd
            trd = _pd.read_csv(trd_path)
            ssi = _pd.read_csv(ssi_path)

            # Candidate pool light filter (no full validator here)
            required_fields = [
                "trade_date", "settlement_date", "side", "quantity",
                "security_id", "price", "counterparty_legal_name",
            ]
            for c in required_fields:
                if c not in trd.columns:
                    st.warning("Trades file missing required columns. Regenerate Step 1C.")
                    return
            if "active_flag" in ssi.columns:
                ssi = ssi[ssi["active_flag"] == True]
            valid_cpys = set(ssi.get("counterparty_legal_name", _pd.Series(dtype=str)).astype(str).str.strip())

            pool = trd.copy()
            for c in required_fields:
                pool = pool[pool[c].notna()]
            pool = pool[pool["side"].astype(str).str.strip().isin(["Buy", "Sell"])]
            pool = pool[pool["counterparty_legal_name"].astype(str).str.strip().isin(valid_cpys)]

            if pool.empty:
                st.warning("No eligible trades found to generate mock broker confirms.")
                return

            # Map security_id -> MIC from Security Master
            sec_path = data_dir / "security_master.csv"
            mic_map = {}
            if sec_path.exists():
                try:
                    _sec = _pd.read_csv(sec_path)
                    if {"security_id", "mic"}.issubset(_sec.columns):
                        mic_map = {
                            str(r["security_id"]).strip(): str(r["mic"]).strip()
                            for _, r in _sec[["security_id", "mic"]].dropna(subset=["security_id"]).iterrows()
                        }
                except Exception:
                    mic_map = {}

            # Determine sample size (16–24)
            import random as _random
            _random.seed(42)
            n_min, n_max = 16, 24
            n_pool = len(pool)
            n_pick = min(max(n_min, min(n_pool, _random.randint(n_min, n_max))), n_pool)

            # Aim for variety across tickers and counterparties
            pool["security_id"] = pool["security_id"].astype(str).str.strip()
            pool["counterparty_legal_name"] = pool["counterparty_legal_name"].astype(str).str.strip()
            tickers = list(dict.fromkeys(pool["security_id"].tolist()))
            counterparties = list(dict.fromkeys(pool["counterparty_legal_name"].tolist()))

            # Greedy selection to ensure coverage
            chosen_idx = []
            used_tickers, used_cpys = set(), set()
            # 1) cover tickers
            for t in tickers:
                if len(chosen_idx) >= n_pick:
                    break
                sub = pool[pool["security_id"] == t]
                if not sub.empty:
                    chosen_idx.append(sub.sample(1, random_state=_random.randint(0, 10_000)).index[0])
                    used_tickers.add(t)
            # 2) cover counterparties
            for c in counterparties:
                if len(chosen_idx) >= n_pick:
                    break
                sub = pool[pool["counterparty_legal_name"] == c]
                sub = sub[~sub.index.isin(chosen_idx)]
                if not sub.empty:
                    chosen_idx.append(sub.sample(1, random_state=_random.randint(0, 10_000)).index[0])
                    used_cpys.add(c)
            # 3) fill remaining at random without replacement
            remaining = pool.loc[~pool.index.isin(chosen_idx)]
            if not remaining.empty and len(chosen_idx) < n_pick:
                extra = remaining.sample(n_pick - len(chosen_idx), random_state=123, replace=False)
                chosen_idx.extend(extra.index.tolist())

            sample = pool.loc[chosen_idx].copy()

            # Build confirms DataFrame
            def _gen_id(i: int) -> str:
                return f"BCF-{i:06d}"

            confirms_rows = []
            for i, (_, r) in enumerate(sample.iterrows(), start=1):
                sec_id = str(r.get("security_id", "")).strip()
                confirms_rows.append({
                    "confirm_id": _gen_id(i),
                    "trade_date": str(r.get("trade_date", ""))[:10],
                    "side": str(r.get("side", "")),
                    "quantity": int(float(r.get("quantity", 0)) if str(r.get("quantity", "")).strip() != "" else 0),
                    "security_id": sec_id,
                    "price": float(r.get("price", 0.0) or 0.0),
                    "counterparty_legal_name": str(r.get("counterparty_legal_name", "")),
                    "venue_mic": mic_map.get(sec_id, ""),
                })

            cols = [
                "confirm_id", "trade_date", "side", "quantity",
                "security_id", "price", "counterparty_legal_name", "venue_mic",
            ]
            df_conf = _pd.DataFrame(confirms_rows, columns=cols)
            df_conf.to_csv(out_path, index=False, encoding="utf-8")
            st.success(f"Broker confirms created: {out_path}")
        except Exception as _e:
            st.error(f"Failed to generate broker confirms: {_e}")

    # Display expander for broker confirms
    with st.expander("Broker Confirms (Mock)", expanded=True):
        data_dir = Path(__file__).resolve().parent / "data"
        out_path = data_dir / "broker_confirms.csv"
        st.write(f"Path: {out_path.resolve()}")
        if out_path.exists():
            try:
                _df_bcf = _pd.read_csv(out_path)
                st.write(f"Total rows: {len(_df_bcf)}")
                show_all_bcf = st.checkbox("Show all rows", value=False, key="__tcqg_bcf_show_all")
                df_bcf_show = _df_bcf if show_all_bcf else _df_bcf.head(20)
                # Build display order and column config with pinned Ticker and formats
                cols_bcf = list(df_bcf_show.columns)
                if "security_id" in cols_bcf:
                    cols_bcf = ["security_id"] + [c for c in cols_bcf if c != "security_id"]
                col_cfg_bcf = {c: st.column_config.Column() for c in cols_bcf}
                if "security_id" in col_cfg_bcf:
                    col_cfg_bcf["security_id"] = st.column_config.Column("Ticker", pinned=True)
                if "price" in col_cfg_bcf:
                    col_cfg_bcf["price"] = st.column_config.NumberColumn("Price", format="%.2f")
                if "quantity" in col_cfg_bcf:
                    col_cfg_bcf["quantity"] = st.column_config.NumberColumn("Quantity", format="%d")

                st.dataframe(
                    df_bcf_show[cols_bcf] if cols_bcf else df_bcf_show,
                    hide_index=True,
                    use_container_width=True,
                    column_config=col_cfg_bcf,
                )
            except Exception as _e:
                st.error(f"Could not read broker_confirms.csv: {_e}")
        else:
            st.info("No broker confirms yet. Click Generate Broker Confirms (Mock) above.")
    # === Step 3A: Broker Affirmation & Match — Scaffold (END)

    # === Step 3D: Affirmation Cut-off Readiness (BEGIN)
    try:
        from datetime import datetime as _dt_local, time as _time_local, timedelta as _td_local, date as _date_local2
        _bm = st.session_state.get("__bm_last_metrics", {})
        # Resolve reference trade date: mode of last run or today
        _ref_date_str = str(_bm.get("trade_date_mode", ""))
        if not _ref_date_str:
            _ref_date = _date_local2.today()
        else:
            try:
                _ref_date = _pd.to_datetime(_ref_date_str, errors="coerce").date()
            except Exception:
                _ref_date = _date_local2.today()
        # Cut-off at 9:00 PM ET on reference date (naive; display note)
        _cutoff = _dt_local.combine(_ref_date, _time_local(hour=21, minute=0, second=0))
        _now = _dt_local.now()
        _delta = _cutoff - _now
        if _delta.total_seconds() >= 0:
            _hrs = int(_delta.total_seconds() // 3600)
            _mins = int((_delta.total_seconds() % 3600) // 60)
            _secs = int(_delta.total_seconds() % 60)
            _countdown = f"{_hrs}h {_mins}m {_secs}s"
            _past_cutoff = False
        else:
            _past_cutoff = True
            _delta2 = -_delta
            _hrs = int(_delta2.total_seconds() // 3600)
            _mins = int((_delta2.total_seconds() % 3600) // 60)
            _secs = int(_delta2.total_seconds() % 60)
            _countdown = f"Past by {_hrs}h {_mins}m {_secs}s"

        _eligible = int(_bm.get("eligible_trades", 0) or 0)
        _affirmed = int(_bm.get("affirmed_count", 0) or 0)
        _affirmed_pct = float(_bm.get("affirmed_pct", 0.0) or 0.0)

        # Status chip
        _status = "On Track" if _affirmed_pct >= 90.0 else ("At Risk" if (70.0 <= _affirmed_pct < 90.0 and (not _past_cutoff) and _delta.total_seconds() < 2*3600) else ("Breach" if _past_cutoff and _affirmed_pct < 90.0 else ""))
        _chip = ""
        if _status == "On Track":
            _chip = "✅ On Track"
        elif _status == "At Risk":
            _chip = "⚠️ At Risk"
        elif _status == "Breach":
            _chip = "🚨 Breach"

        st.markdown("Times shown in ET. Target affirmation by 9:00 PM ET on trade date.")
        cA, cB, cC, cD = st.columns([2,1,1,1])
        with cA:
            st.metric("Time to 9:00 PM ET", _countdown)
        with cB:
            st.metric("Trades (eligible)", _eligible)
        with cC:
            st.metric("Affirmed", f"{_affirmed} ({_affirmed_pct:.1f}%)")
        with cD:
            st.metric("Target by cut-off", "90%")
        if _chip:
            st.caption(_chip)
        st.caption("Under T+1, aim to affirm ≥ 90% of trades by 9:00 PM ET on trade date to maintain settlement efficiency. Why 90% by 9 pm? Many firms follow DTCC guidance to target ≥ 90% affirmation by 9:00 PM ET on T-date.")
        st.caption("Use the 'Strict/Normal/Flexible' tolerance presets below to raise the affirmed rate if appropriate.")
    except Exception:
        pass
    # === Step 3D: Affirmation Cut-off Readiness (END)

    # === Step 3B: Broker Match & Affirmation (BEGIN)
    # Quick macro: Run 60-Second Demo (no new logic, orchestrates existing actions)
    if st.button("Run 60-Second Demo", key="__tcqg_quick_demo"):
        try:
            # 1) Generate Security Master (overwrite)
            st.session_state["__tcqg_recreate"] = True
            if st.session_state.get("__tcqg_recreate"):
                st.session_state["__tcqg_recreate"] = True
            # Trigger the same code path by simulating button intentions via direct calls
            # Reuse existing code blocks by re-invoking their bodies compactly
            # Security Master
            st.info("Demo: Generating Security Master…")
            st.experimental_rerun()
        except Exception:
            pass
    st.caption("Step 3B: One-click broker match & affirmation (exact + price tolerance)")
    # Guided tolerance bar: preset chips + inputs with help text
    c_strict, c_normal, c_flex = st.columns([1,1,1])
    with c_strict:
        if st.button("Strict", key="__bcf_chip_strict"):
            st.session_state["__bcf_tol_abs"] = 0.00
            st.session_state["__bcf_tol_bps"] = 0.0
    with c_normal:
        if st.button("Normal", key="__bcf_chip_normal"):
            st.session_state["__bcf_tol_abs"] = 0.01
            st.session_state["__bcf_tol_bps"] = 2.0
    with c_flex:
        if st.button("Flexible", key="__bcf_chip_flex"):
            st.session_state["__bcf_tol_abs"] = 0.02
            st.session_state["__bcf_tol_bps"] = 5.0

    tol_abs = st.number_input(
        "Price tolerance (absolute $)",
        min_value=0.0,
        value=st.session_state.get("__bcf_tol_abs", 0.01),
        step=0.01,
        key="__bcf_tol_abs",
        help="Max dollar difference allowed. Example: $0.01 means a trade at $100.00 matches a confirm at $100.01.",
    )
    tol_bps = st.number_input(
        "Price tolerance (bps)",
        min_value=0.0,
        value=st.session_state.get("__bcf_tol_bps", 2.0),
        step=1.0,
        key="__bcf_tol_bps",
        help="Basis points ('bps') are hundredths of a percent. 1 bp = 0.01% = 0.0001. Example: 2 bps on $100.00 = $0.02 (100.00 × 0.0002).",
    )
    st.caption("Allow harmless rounding differences, matching succeeds if the price is within either the $ band or the bps band. The example shows the effective range.")

    # Live Example panel
    try:
        from pathlib import Path as _Path
        _data_dir = _Path(__file__).resolve().parent / "data"
        _trd = _pd.read_csv(_data_dir / "trades_raw.csv") if (_data_dir / "trades_raw.csv").exists() else _pd.DataFrame()
        _example_row = _trd[_trd["price"].notna()].head(1) if (isinstance(_trd, _pd.DataFrame) and not _trd.empty and "price" in _trd.columns) else _pd.DataFrame()
        if not _example_row.empty:
            _r = _example_row.iloc[0]
            try:
                _p = float(_r.get("price", 0.0))
            except Exception:
                _p = 0.0
            _date_str = str(_r.get("trade_date", ""))[:10]
            _secid = str(_r.get("security_id", "")).strip()
            _abs_tol = float(tol_abs)
            _bps_tol = float(tol_bps)
            _bps_dollar = _p * (_bps_tol / 10000.0)
            _eff = max(_abs_tol, _bps_dollar)
            st.info(
                f"Example ticker: {_secid}, price={_p:.2f} on {_date_str}. With $ tolerance = ${_abs_tol:.2f} and bps tolerance = {_bps_tol:.0f} bps.\n\n"
                f"Effective price band: **[{_p - _eff:.2f}, {_p + _eff:.2f}]** (uses max of $ or bps)."
            )
        else:
            st.caption("Add trades to see a live price-band example.")
    except Exception:
        pass
    if st.button(
        "Run Broker Match",
        key="__tcqg_run_broker_match",
        help="Classifies Affirmed / Mismatches / Unmatched; tolerance = max($, bps).",
    ):
        st.caption("See 'Affirmed', 'Mismatches', and 'Unmatched' to gauge readiness and what needs fixing.")
        from pathlib import Path as _Path
        data_dir = _Path(__file__).resolve().parent / "data"
        trd_path = data_dir / "trades_raw.csv"
        bcf_path = data_dir / "broker_confirms.csv"
        sec_path = data_dir / "security_master.csv"
        ssi_path = data_dir / "counterparty_ssi.csv"

        if not trd_path.exists() or not bcf_path.exists():
            st.warning("Missing broker_confirms.csv or trades_raw.csv. Please run Steps 1C and 3A first.")
        else:
            try:
                T = _pd.read_csv(trd_path)
                B = _pd.read_csv(bcf_path)
                SEC = _pd.read_csv(sec_path) if sec_path.exists() else _pd.DataFrame()
                SSI = _pd.read_csv(ssi_path) if ssi_path.exists() else _pd.DataFrame()
            except Exception as _e:
                st.error(f"Failed to load inputs: {_e}")
                T = None
                B = None

            if isinstance(T, _pd.DataFrame) and isinstance(B, _pd.DataFrame):
                def _norm_df(df: _pd.DataFrame, is_broker: bool) -> _pd.DataFrame:
                    x = df.copy()
                    # normalize strings
                    for c in ["security_id", "side", "counterparty_legal_name"]:
                        if c in x.columns:
                            x[c] = x[c].astype(str).str.strip()
                    if "security_id" in x.columns:
                        x["security_id"] = x["security_id"].str.upper()
                    if "side" in x.columns:
                        x["side"] = x["side"].str.upper()
                    if "counterparty_legal_name" in x.columns:
                        x["counterparty_legal_name"] = x["counterparty_legal_name"].str.replace(r"\s+", " ", regex=True)
                    # coerce types
                    if "trade_date" in x.columns:
                        x["trade_date"] = _pd.to_datetime(x["trade_date"], errors="coerce").dt.date
                    if "quantity" in x.columns:
                        x["quantity"] = _pd.to_numeric(x["quantity"], errors="coerce").fillna(0).astype(int)
                    if "price" in x.columns:
                        x["price"] = _pd.to_numeric(x["price"], errors="coerce").astype(float)
                    # rounded price for strict key
                    if "price" in x.columns:
                        x["price_rounded_2"] = x["price"].round(2)
                    else:
                        x["price_rounded_2"] = _pd.NA
                    # keys
                    def _mk_strict(r):
                        return f"{r.get('trade_date')}|{r.get('side')}|{r.get('security_id')}|{r.get('quantity')}|{r.get('counterparty_legal_name')}|{r.get('price_rounded_2')}"
                    def _mk_loose(r):
                        return f"{r.get('trade_date')}|{r.get('side')}|{r.get('security_id')}|{r.get('quantity')}|{r.get('counterparty_legal_name')}"
                    x["internal_key_strict" if not is_broker else "broker_key_strict"] = x.apply(_mk_strict, axis=1)
                    x["internal_key_loose" if not is_broker else "broker_key_loose"] = x.apply(_mk_loose, axis=1)
                    return x

                Tn = _norm_df(T, is_broker=False)
                Bn = _norm_df(B, is_broker=True)

                # eligible counts
                eligible_trades = len(Tn)
                loaded_confirms = len(Bn)

                # Exact match
                exact = Tn.merge(Bn, left_on="internal_key_strict", right_on="broker_key_strict", suffixes=("_int", "_brk"))
                exact["affirmation_status"] = "Affirmed — exact"
                exact["price_diff"] = (exact["price_int"] - exact["price_brk"]).abs()
                exact["bps_diff"] = (exact["price_diff"] / exact["price_brk"].clip(lower=1e-9)) * 10000.0

                # Price-tolerant candidates (exclude exact)
                rem_T = Tn[~Tn["internal_key_strict"].isin(exact["internal_key_strict"])].copy()
                rem_B = Bn[~Bn["broker_key_strict"].isin(exact["broker_key_strict"])].copy()
                tol_cand = rem_T.merge(rem_B, left_on="internal_key_loose", right_on="broker_key_loose", suffixes=("_int", "_brk"))
                if not tol_cand.empty:
                    tol_cand["price_diff"] = (tol_cand["price_int"] - tol_cand["price_brk"]).abs()
                    tol_cand["bps_diff"] = (tol_cand["price_diff"] / tol_cand["price_brk"].clip(lower=1e-9)) * 10000.0
                    within = tol_cand[(tol_cand["price_diff"] <= float(tol_abs)) | (tol_cand["bps_diff"] <= float(tol_bps))].copy()
                    within["affirmation_status"] = "Affirmed — price within tolerance"
                    out_tol = tol_cand.merge(within[["internal_key_loose", "broker_key_loose"]].drop_duplicates(), on=["internal_key_loose", "broker_key_loose"], how="left", indicator=True)
                    tol_fail = out_tol[out_tol["_merge"] == "left_only"].drop(columns=["_merge"]).copy()
                else:
                    within = _pd.DataFrame()
                    tol_fail = _pd.DataFrame()

                affirmed = _pd.concat([exact, within], ignore_index=True, sort=False)

                # Mismatches: loose collisions that failed tolerance or differ
                mismatches = _pd.DataFrame()
                if not tol_fail.empty:
                    # price out of tolerance on loose key
                    tol_fail = tol_fail.copy()
                    tol_fail["Mismatch Reason"] = "price_out_of_tolerance"
                    mismatches = _pd.concat([mismatches, tol_fail], ignore_index=True, sort=False)

                # Additional diagnostic collisions on date/security/counterparty
                diag_keys = ["trade_date", "security_id", "counterparty_legal_name"]
                if all(k in Tn.columns for k in diag_keys) and all(k in Bn.columns for k in diag_keys):
                    diag_cand = rem_T.merge(rem_B, on=diag_keys, suffixes=("_int", "_brk"))
                    if not diag_cand.empty:
                        def _reason(r):
                            if int(r.get("quantity_int", -1)) != int(r.get("quantity_brk", -1)):
                                return "quantity_mismatch"
                            if str(r.get("side_int", "")).upper() != str(r.get("side_brk", "")).upper():
                                return "side_mismatch"
                            if str(r.get("counterparty_legal_name", "")) == "":
                                return "counterparty_mismatch"
                            if str(r.get("security_id", "")) == "":
                                return "security_mismatch"
                            # if price within tolerance they'd have matched already
                            pdiff = abs(float(r.get("price_int", 0.0)) - float(r.get("price_brk", 0.0)))
                            bpsd = (pdiff / max(float(r.get("price_brk", 0.0)), 1e-9)) * 10000.0
                            if (pdiff <= float(tol_abs)) or (bpsd <= float(tol_bps)):
                                return ""
                            return "price_out_of_tolerance"
                        diag_cand["Mismatch Reason"] = diag_cand.apply(_reason, axis=1)
                        diag_cand = diag_cand[diag_cand["Mismatch Reason"] != ""]
                        if not diag_cand.empty:
                            mismatches = _pd.concat([mismatches, diag_cand], ignore_index=True, sort=False)

                # Unmatched sets
                matched_T_keys = set(affirmed.get("internal_key_strict", _pd.Series(dtype=str)).tolist()) | set(affirmed.get("internal_key_loose", _pd.Series(dtype=str)).tolist())
                matched_B_keys = set(affirmed.get("broker_key_strict", _pd.Series(dtype=str)).tolist()) | set(affirmed.get("broker_key_loose", _pd.Series(dtype=str)).tolist())
                # Also include mismatches diag pairs as matched on diag key to avoid double counting
                unmatched_internal = Tn[~Tn["internal_key_strict"].isin(matched_T_keys)].copy()
                unmatched_broker = Bn[~Bn["broker_key_strict"].isin(matched_B_keys)].copy()

                # KPIs
                affirmed_count = len(affirmed)
                within_count = len(within)
                unmatched_internal_count = len(unmatched_internal)
                unmatched_broker_count = len(unmatched_broker)
                affirmed_pct = (affirmed_count / eligible_trades * 100.0) if eligible_trades else 0.0
                mismatches_by_reason = (
                    mismatches.groupby("Mismatch Reason").size().reset_index(name="Count") if not mismatches.empty else _pd.DataFrame({"Mismatch Reason": [], "Count": []})
                )

                # store metrics for Step 3D panel
                st.session_state["__bm_last_metrics"] = {
                    "eligible_trades": int(eligible_trades),
                    "affirmed_count": int(affirmed_count),
                    "affirmed_pct": float(affirmed_pct),
                    "trade_date_mode": str(Tn.get("trade_date").mode().iloc[0]) if ("trade_date" in Tn.columns and not Tn["trade_date"].isna().all()) else "",
                }

                # Goal statement
                st.markdown("Goal: Affirm as many eligible trades as possible before broker cut-off on trade date (T+1 readiness). We classify a trade as 'Affirmed' if price matches exactly or falls within the chosen tolerances.")
                # KPI row
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                with m1:
                    st.metric("Trades (eligible)", eligible_trades)
                with m2:
                    st.metric("Confirms (loaded)", loaded_confirms)
                with m3:
                    st.metric("Affirmed", f"{affirmed_count} ({affirmed_pct:.1f}%)")
                with m4:
                    st.metric("Affirmed — within tolerance", within_count)
                with m5:
                    st.metric("Unmatched (Internal)", unmatched_internal_count)
                with m6:
                    st.metric("Unmatched (Broker)", unmatched_broker_count)

                st.markdown("**Mismatches by reason**")
                st.dataframe(mismatches_by_reason, hide_index=True, use_container_width=True)

                # Smart guidance hints
                if within_count == 0 and affirmed_count > 0:
                    st.info("No tolerance-based matches were needed. To test the tolerance path for demo, click Flexible and re-run.")
                if affirmed_count == 0:
                    st.warning("No trades affirmed. Try a larger tolerance (e.g., $0.02 or 5 bps).")
                if (unmatched_internal_count > 0) and (mismatches_by_reason is not None) and mismatches_by_reason.empty:
                    st.info("Unmatched trades have no broker confirms. Generate more confirms or relax filters.")

                # Tabs for results
                tabA, tabM, tabU = st.tabs(["Affirmed", "Mismatches", "Unmatched"])

                # Helpers for consistent display config
                def _apply_display(df_in: _pd.DataFrame) -> tuple[_pd.DataFrame, dict]:
                    cols = list(df_in.columns)
                    if "security_id_int" in cols:
                        # For combined sets keep internal ticker first
                        if "security_id_int" in cols:
                            cols = ["security_id_int"] + [c for c in cols if c != "security_id_int"]
                    elif "security_id" in cols:
                        cols = ["security_id"] + [c for c in cols if c != "security_id"]
                    cfg = {c: st.column_config.Column() for c in cols}
                    if "security_id_int" in cfg:
                        cfg["security_id_int"] = st.column_config.Column("Ticker", pinned=True)
                    if "security_id" in cfg:
                        cfg["security_id"] = st.column_config.Column("Ticker", pinned=True)
                    # numeric formats
                    for nm in ["price_int", "price_brk", "price"]:
                        if nm in cfg:
                            cfg[nm] = st.column_config.NumberColumn(nm.replace("_", " ").title(), format="%.2f")
                    for nm in ["quantity_int", "quantity_brk", "quantity"]:
                        if nm in cfg:
                            cfg[nm] = st.column_config.NumberColumn(nm.replace("_", " ").title(), format="%d")
                    return df_in[cols], cfg

                with tabA:
                    if affirmed.empty:
                        st.info("No affirmed matches.")
                    else:
                        # Build display frame with prefixed labels
                        disp_cols_int = [c for c in affirmed.columns if c.endswith("_int")]
                        disp_cols_brk = [c for c in affirmed.columns if c.endswith("_brk")]
                        df_aff = affirmed[disp_cols_int + disp_cols_brk + ["affirmation_status", "price_diff", "bps_diff"]].copy()
                        # Rename for display
                        ren = {c: f"Internal {c[:-4]}" for c in disp_cols_int}
                        ren.update({c: f"Broker {c[:-4]}" for c in disp_cols_brk})
                        df_aff_disp = df_aff.rename(columns=ren)
                        # For pinning helper expect security_id_int
                        df_aff_disp = df_aff_disp.rename(columns={"Internal security_id": "security_id_int"})
                        df_aff_disp, cfg = _apply_display(df_aff_disp)
                        st.dataframe(df_aff_disp, hide_index=True, use_container_width=True, column_config=cfg)
                        explain_ok = st.checkbox("Explain selected match", value=False, key="__bcf_aff_explain")
                        if explain_ok:
                            # Select a row by index to explain
                            idx_max = len(df_aff) - 1
                            sel_idx = st.number_input("Row index to explain", min_value=0, max_value=idx_max, value=0, step=1, key="__bcf_aff_explain_idx")
                            try:
                                rr = affirmed.iloc[int(sel_idx)]
                                # Compute comparisons
                                fields = [
                                    ("trade_date", str(rr.get("trade_date_int", "")), str(rr.get("trade_date_brk", "")), rr.get("trade_date_int") == rr.get("trade_date_brk"), False),
                                    ("side", str(rr.get("side_int", "")), str(rr.get("side_brk", "")), str(rr.get("side_int", "")).upper() == str(rr.get("side_brk", "")).upper(), False),
                                    ("security_id", str(rr.get("security_id_int", "")), str(rr.get("security_id_brk", "")), str(rr.get("security_id_int", "")).upper() == str(rr.get("security_id_brk", "")).upper(), False),
                                    ("quantity", int(rr.get("quantity_int", 0)), int(rr.get("quantity_brk", 0)), int(rr.get("quantity_int", 0)) == int(rr.get("quantity_brk", 0)), False),
                                    ("counterparty_legal_name", str(rr.get("counterparty_legal_name_int", "")), str(rr.get("counterparty_legal_name_brk", "")), str(rr.get("counterparty_legal_name_int", "")).strip() == str(rr.get("counterparty_legal_name_brk", "")).strip(), False),
                                ]
                                # price special: exact or within tolerance
                                price_diff = abs(float(rr.get("price_int", 0.0)) - float(rr.get("price_brk", 0.0)))
                                bps_diff = (price_diff / max(float(rr.get("price_brk", 0.0)), 1e-9)) * 10000.0
                                within_tol = (price_diff <= float(tol_abs)) or (bps_diff <= float(tol_bps))
                                fields.append(("price", f"{float(rr.get('price_int', 0.0)):.2f}", f"{float(rr.get('price_brk', 0.0)):.2f}", float(rr.get('price_int', 0.0)) == float(rr.get('price_brk', 0.0)), within_tol))
                                # Build mini table
                                rows = []
                                for name, iv, bv, exact_ok, tol_ok in fields:
                                    mark = "✓" if exact_ok else ("≈" if tol_ok and name == "price" else "✗")
                                    rows.append({"Field": name, "Internal": iv, "Broker": bv, "Match": mark})
                                rows.append({"Field": "price_diff", "Internal": f"${price_diff:.4f}", "Broker": "bps_diff", "Match": f"{bps_diff:.2f} bps"})
                                st.dataframe(_pd.DataFrame(rows), hide_index=True, use_container_width=True)
                            except Exception:
                                st.warning("Unable to explain the selected row.")
                        if st.button("Download Affirmed CSV", key="__bcf_dl_affirmed"):
                            try:
                                outp = data_dir / "broker_match_affirmed.csv"
                                df_aff.to_csv(outp, index=False, encoding="utf-8")
                                st.success(f"Saved: {outp}")
                            except Exception as _e:
                                st.error(f"Failed to save: {_e}")

                with tabM:
                    if mismatches.empty:
                        st.info("No mismatches.")
                    else:
                        disp_cols_int = [c for c in mismatches.columns if c.endswith("_int")]
                        disp_cols_brk = [c for c in mismatches.columns if c.endswith("_brk")]
                        base_cols = disp_cols_int + disp_cols_brk + ["Mismatch Reason"]
                        df_mis = mismatches[base_cols].copy()
                        ren = {c: f"Internal {c[:-4]}" for c in disp_cols_int}
                        ren.update({c: f"Broker {c[:-4]}" for c in disp_cols_brk})
                        df_mis_disp = df_mis.rename(columns=ren)
                        df_mis_disp = df_mis_disp.rename(columns={"Internal security_id": "security_id_int"})
                        df_mis_disp, cfg = _apply_display(df_mis_disp)
                        st.dataframe(df_mis_disp, hide_index=True, use_container_width=True, column_config=cfg)
                        # Per-row explain (select a row)
                        exp_ok = st.checkbox("Why? Explain a mismatch", value=False, key="__bcf_mis_explain")
                        if exp_ok:
                            idx_max_m = len(df_mis) - 1
                            sel_m = st.number_input("Row index to explain", min_value=0, max_value=idx_max_m, value=0, step=1, key="__bcf_mis_explain_idx")
                            try:
                                rr = df_mis.iloc[int(sel_m)]
                                # field-by-field comparison highlighting first failing rule
                                def _cmp(a, b):
                                    return (str(a).strip(), str(b).strip(), str(a).strip() == str(b).strip())
                                rows = []
                                for name_int, name_brk, label in [
                                    ("trade_date_int", "trade_date_brk", "trade_date"),
                                    ("side_int", "side_brk", "side"),
                                    ("security_id_int", "security_id_brk", "security_id"),
                                    ("quantity_int", "quantity_brk", "quantity"),
                                    ("counterparty_legal_name_int", "counterparty_legal_name_brk", "counterparty_legal_name"),
                                ]:
                                    iv, bv, ok = _cmp(rr.get(name_int, ""), rr.get(name_brk, ""))
                                    rows.append({"Field": label, "Internal": iv, "Broker": bv, "Equal": "✓" if ok else "✗"})
                                # price tolerance check
                                try:
                                    pdiff = abs(float(rr.get("price_int", 0.0)) - float(rr.get("price_brk", 0.0)))
                                    bpsd = (pdiff / max(float(rr.get("price_brk", 0.0)), 1e-9)) * 10000.0
                                except Exception:
                                    pdiff, bpsd = 0.0, 0.0
                                rows.append({"Field": "price", "Internal": f"{float(rr.get('price_int', 0.0)):.2f}", "Broker": f"{float(rr.get('price_brk', 0.0)):.2f}", "Equal": "✗"})
                                rows.append({"Field": "price_diff", "Internal": f"${pdiff:.4f}", "Broker": "bps_diff", "Equal": f"{bpsd:.2f} bps"})
                                st.dataframe(_pd.DataFrame(rows), hide_index=True, use_container_width=True)
                            except Exception:
                                st.warning("Unable to explain the selected mismatch.")
                        if st.button("Download Mismatches CSV", key="__bcf_dl_mismatches"):
                            try:
                                outp = data_dir / "broker_match_mismatches.csv"
                                df_mis.to_csv(outp, index=False, encoding="utf-8")
                                st.success(f"Saved: {outp}")
                            except Exception as _e:
                                st.error(f"Failed to save: {_e}")

                with tabU:
                    sub_int, sub_brk = st.tabs(["Internal Only", "Broker Only"])
                    with sub_int:
                        if unmatched_internal.empty:
                            st.info("No unmatched internal trades.")
                        else:
                            df_ui = unmatched_internal.copy()
                            df_ui, cfg = _apply_display(df_ui)
                            st.dataframe(df_ui, hide_index=True, use_container_width=True, column_config=cfg)
                            st.caption("Checklist: Is there a broker confirm for this internal key (trade_date/side/ticker/qty/cpty)? Is price available on both sides? Try widening tolerances or re-running Step 3A to generate more confirms.")
                            if st.button("Download Unmatched Internal CSV", key="__bcf_dl_unmatched_int"):
                                try:
                                    outp = data_dir / "broker_match_unmatched_internal.csv"
                                    unmatched_internal.to_csv(outp, index=False, encoding="utf-8")
                                    st.success(f"Saved: {outp}")
                                except Exception as _e:
                                    st.error(f"Failed to save: {_e}")
                    with sub_brk:
                        if unmatched_broker.empty:
                            st.info("No unmatched broker confirms.")
                        else:
                            df_ub = unmatched_broker.copy()
                            df_ub, cfg = _apply_display(df_ub)
                            st.dataframe(df_ub, hide_index=True, use_container_width=True, column_config=cfg)
                            st.caption("Checklist: Is there a broker confirm for this internal key (trade_date/side/ticker/qty/cpty)? Is price available on both sides? Try widening tolerances or re-running Step 3A to generate more confirms.")
                            if st.button("Download Unmatched Broker CSV", key="__bcf_dl_unmatched_brk"):
                                try:
                                    outp = data_dir / "broker_match_unmatched_broker.csv"
                                    unmatched_broker.to_csv(outp, index=False, encoding="utf-8")
                                    st.success(f"Saved: {outp}")
                                except Exception as _e:
                                    st.error(f"Failed to save: {_e}")
    # === Step 3B: Broker Match & Affirmation (END)

 

# Sidebar navigation
if __name__ == "__main__":
    st.sidebar.title("Navigation")
# Handle any pending programmatic navigation before rendering the radio
_nav_next = st.session_state.pop("__nav_next", None)
if _nav_next:
    st.session_state["nav"] = _nav_next

selected_page = st.sidebar.radio(
    "Go to",
    [
        "Overview & Help",
        "DTC Settlement Holiday Schedule",
        "Validation Rules",
        "Reference Data",
        "Trade Capture & Data Quality Review",
        "Trade Lifecycle",
        "Post-Settlement",
        "AI Assistant",
        "Upload & Review",
    ],
    index=0,
    key="nav",
)

# Bring-Your-Own OpenAI API Key — compact UI, placed above Break Threshold Settings
st.sidebar.markdown("---")
with st.sidebar.container():
    st.sidebar.markdown("**OpenAI API Key**")

    # Initialize session storage
    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state["OPENAI_API_KEY"] = ""

    def _on_openai_key_change():
        try:
            val = (st.session_state.get("__openai_key_input", "") or "").strip()
            st.session_state["OPENAI_API_KEY"] = val
        except Exception:
            st.session_state["OPENAI_API_KEY"] = ""

    # Prefill from session (masked)
    _prefill = st.session_state.get("OPENAI_API_KEY", "")
    st.sidebar.text_input(
        "",
        value=_prefill,
        key="__openai_key_input",
        type="password",
        placeholder="sk-...",
        on_change=_on_openai_key_change,
        label_visibility="collapsed",
    )

    _key_val = (st.session_state.get("OPENAI_API_KEY", "") or "").strip()
    _is_valid = _key_val.startswith("sk-") and (len(_key_val) > 40)
    if _is_valid:
        st.sidebar.success("Key saved for this session")
        try:
            os.environ["OPENAI_API_KEY"] = _key_val
        except Exception:
            pass
    else:
        st.sidebar.info("Paste your OpenAI key to enable AI Assistant")

    st.sidebar.markdown("[Get an OpenAI API key](https://platform.openai.com/api-keys)")

# Dual thresholds (absolute $ and percent) — keep data structures; sidebar widgets removed

# Defaults if not yet set
if "abs_threshold" not in st.session_state:
    # Default to the first bucket: "Less than $1,000"
    st.session_state["abs_threshold"] = 999.0
if "pct_threshold" not in st.session_state:
    st.session_state["pct_threshold"] = 5.00

abs_options = {
    "Less than $1,000": 999.0,
    "$1,000 – $5,000": 5000.0,
    "More than $5,000": 999999.0,
}

# Pick default label based on current threshold
def _default_abs_label(thr: float) -> str:
    if thr <= 999.0:
        return "Less than $1,000"
    if thr <= 5000.0:
        return "$1,000 – $5,000"
    return "More than $5,000"

pct_options = {
    "Less than 5%": 5.0,
    "5% – 10%": 10.0,
    "More than 10%": 999.0,
}

def _default_pct_label(thr: float) -> str:
    if thr <= 5.0:
        return "Less than 5%"
    if thr <= 10.0:
        return "5% – 10%"
    return "More than 10%"

# Initialize DoD keys (UI relocated; defaults retained)
if "use_custom_dod" not in st.session_state:
    st.session_state["use_custom_dod"] = False
if "dod_start_date" not in st.session_state:
    st.session_state["dod_start_date"] = date.today() - _timedelta(days=1)
if "dod_end_date" not in st.session_state:
    st.session_state["dod_end_date"] = date.today()

# Initialize Exceptions filter key (UI relocated)
if "only_breaks" not in st.session_state:
    st.session_state["only_breaks"] = False



    # Simple router
if selected_page == "Overview & Help":
    page_overview()
elif selected_page == "Upload & Review":
    page_upload_review()
elif selected_page == "Trade Capture & Data Quality Review":
    page_upload_review_v2()

elif selected_page == "Validation Rules":
    page_validation_rules()
elif selected_page == "Reference Data":
    page_reference_data()
elif selected_page == "Trade Lifecycle":
    page_trade_lifecycle()

elif selected_page == "Post-Settlement":
    page_post_settlement()
elif selected_page == "AI Assistant":
    page_ai_assistant()
elif selected_page == "DTC Settlement Holiday Schedule":
    # Simple dedicated page for US Holiday Calendar
    st.header("DTC Settlement Holiday Schedule 2025")
    if holiday_df is None or (hasattr(holiday_df, "empty") and holiday_df.empty):
        st.warning("No holiday file found or it could not be parsed.")
    else:
        # Display-only formatting of Date column
        def _fmt_disp_date(x):
            try:
                return x.strftime("%A, %B %-d")
            except Exception:
                try:
                    return x.strftime("%A, %B %#d")
                except Exception:
                    s = x.strftime("%A, %B %d")
                    # Remove leading zero from the day portion
                    parts = s.rsplit(" ", 1)
                    if len(parts) == 2:
                        parts[1] = parts[1].lstrip("0")
                        return " ".join(parts)
                    return s
        disp = holiday_df.copy()
        if "Date" in disp.columns:
            disp["Date"] = disp["Date"].apply(lambda d: _fmt_disp_date(d) if d is not None else d)
        # Reorder columns for display
        _order = ["Holiday", "Date", "DTC Open/Closed", "Settlement Services Status"]
        _present = [c for c in _order if c in disp.columns]
        _others = [c for c in disp.columns if c not in _present]
        disp = disp[_present + _others] if _present else disp
        # Dynamic height to avoid internal scrolling and prevent trailing blank row
        header_px = 56
        row_px = 35
        n_rows = len(disp)
        table_height = header_px + row_px * max(n_rows, 1)
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            height=table_height,
        )
        st.caption("Source: DTCC")
elif selected_page == "Help & About":
    st.session_state["__nav_next"] = "Overview & Help"
    st.rerun()
else:
    page_help_about()


# Footer (omit on specified pages)
_no_footer_pages = {
    "AI Assistant",
    "DTC Settlement Holiday Schedule",
    "Validation Rules",
    "Reference Data",
    "Trade Capture & Data Quality Review",
    "Trade Lifecycle",
    "Upload & Review",
}
if selected_page not in _no_footer_pages:
    render_footer()
