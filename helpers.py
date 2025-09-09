import pandas as _pd
import numpy as _np

def _fmt_money(v):
    try:
        return "" if v is None or (_pd.isna(v)) else f"{float(v):,.2f}"
    except Exception:
        return ""

def _fmt_money_signed(v):
    try:
        return "" if v is None or (_pd.isna(v)) else f"{float(v):+,.2f}"
    except Exception:
        return ""

def _fmt_qty(v):
    try:
        return "" if v is None or (_pd.isna(v)) else f"{int(float(v)):,}"
    except Exception:
        return ""

def _fmt_pct_signed(x):
    """Return signed percent with 2 decimals; '—%' for non-finite."""
    try:
        import math
        if x is None:
            return "—%"
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "—%"
        return f"{v:+.2f}%"
    except Exception:
        return "—%"

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

    qty = _pd.to_numeric(df.get("quantity_num"), errors="coerce")
    unit_cost = _pd.to_numeric(df.get("unit_cost_num"), errors="coerce")
    price = _pd.to_numeric(
        df.get("market_price_num") if "market_price_num" in df.columns else df.get("market_price"),
        errors="coerce",
    )
    cost_basis = _pd.to_numeric(df.get("cost_basis"), errors="coerce")
    abs_delta = _pd.to_numeric(df.get("unrealized_gain_loss"), errors="coerce")
    pct_delta = _pd.to_numeric(df.get("unrealized_gain_loss_pct"), errors="coerce")
    if "unrealized_gain_loss_pct" not in df.columns or (isinstance(pct_delta, _pd.Series) and pct_delta.isna().all()):
        pct_delta = _np.where(
            (cost_basis > 0) & (~abs_delta.isna()),
            (abs_delta / cost_basis) * 100.0,
            _np.nan,
        )

    flags, reasons, cats = [], [], []
    for i in range(len(df)):
        a = abs_delta.iloc[i] if not _pd.isna(abs_delta.iloc[i]) else _np.nan
        p = pct_delta[i] if not isinstance(pct_delta, _pd.Series) else pct_delta.iloc[i]
        q = qty.iloc[i] if len(qty) > i else _np.nan
        uc = unit_cost.iloc[i] if len(unit_cost) > i else _np.nan
        pr = price.iloc[i] if len(price) > i else _np.nan

        if _pd.isna(q) or _pd.isna(uc):
            flags.append("⚠️"); reasons.append("Missing quantity or unit cost"); cats.append("Data Missing"); continue
        if _pd.isna(pr):
            flags.append("🚨"); reasons.append("No market price available"); cats.append("No Price Data"); continue
        if _pd.isna(a) or _pd.isna(p):
            flags.append("⚠️"); reasons.append("Missing P/L values"); cats.append("Data Missing"); continue

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

def build_enriched_display_frame(df_numeric: _pd.DataFrame) -> _pd.DataFrame:
    """
    Create a display-ready DataFrame with formatting applied.
    Preserves your desired column order and uses shared formatters.
    """
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
        # Include if present
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
        "Market Value",
        "Unrealized Gain/Loss",
        "Unrealized Gain/Loss %",
        "Day Value Change",
        "Day Value Change %",
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

    final_cols = [c for c in desired_order if c in df_disp.columns]
    return df_disp[final_cols]

