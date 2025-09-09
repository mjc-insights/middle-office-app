import streamlit as st
import hashlib
import io

def page_ai_assistant():
    from ai_assistant import show_ai_assistant_page, resolve_openai_key
    key = (resolve_openai_key() or "").strip()
    if not key:
        st.info(
            "Paste your OpenAI API key in the sidebar to enable the assistant. "
            "[Get an OpenAI API key](https://platform.openai.com/api-keys)"
        )
        return
    show_ai_assistant_page()

def compute_lifecycle_view(trades_df, holidays: set[str], as_of):
    """Return lifecycle view dict with processed dataframe, KPIs, and groupings.

    Inputs are read-only. No I/O side-effects.
    """
    import pandas as pd
    from datetime import date as _date, timedelta as _td, datetime as _dt
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        ZoneInfo = None  # type: ignore

    # Derive as_of if not provided or invalid
    if not isinstance(as_of, _date):
        try:
            as_of = (_dt.now(ZoneInfo("America/New_York")) if ZoneInfo else _dt.utcnow()).date()
        except Exception:
            as_of = _dt.utcnow().date()

    # Defensive copy
    df = trades_df.copy() if hasattr(trades_df, "copy") else None
    if df is None:
        return {"df_all": pd.DataFrame(), "kpis": {}, "by_counterparty": pd.DataFrame(), "by_security": pd.DataFrame()}

    # Normalize dates
    for col in ["trade_date", "settlement_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # Drop rows with invalid settlement_date
    if "settlement_date" not in df.columns:
        df = df.iloc[0:0].copy()
    else:
        df = df[df["settlement_date"].notna()].copy()

    # Business day helper (weekdays only for v1)
    def add_business_days(d: _date, n: int) -> _date:
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        cur = d
        while remaining > 0:
            cur = cur + _td(days=step)
            if cur.weekday() < 5:
                remaining -= 1
        return cur

    # Weekend/holiday flag helper (prefer app.weekend_or_holiday_flag)
    def _is_weekend_or_holiday(sd_val) -> bool:
        try:
            from app import weekend_or_holiday_flag as _woh
            flag, _ = _woh(sd_val, holidays)
            return bool(flag)
        except Exception:
            try:
                d = pd.to_datetime(sd_val, errors="coerce").date()
            except Exception:
                return False
            if d is None:
                return False
            return d.weekday() >= 5

    # Status derivation
    due_today = as_of
    due_tomorrow = add_business_days(as_of, 1)

    def _status_for(sd: _date) -> str:
        if sd == due_today:
            return "Due Today"
        if sd == due_tomorrow:
            return "Due Tomorrow"
        if sd < due_today:
            return "Past-Due"
        return "Future"

    # STP readiness flag
    if "stp_ready" in df.columns:
        stp_ready_series = df["stp_ready"].astype(bool)
    elif "stp_ready_flag" in df.columns:
        stp_ready_series = df["stp_ready_flag"].astype(bool)
    elif ("Rule Code" in df.columns) or ("Exception Reason" in df.columns):
        stp_ready_series = pd.Series(False, index=df.index)
    else:
        stp_ready_series = pd.Series(True, index=df.index)

    # Exposure
    q = pd.to_numeric(df.get("quantity"), errors="coerce") if "quantity" in df.columns else 0
    p = pd.to_numeric(df.get("price"), errors="coerce") if "price" in df.columns else 0
    exposure_abs = (q * p).abs().fillna(0.0) if hasattr(q, "__mul__") else 0.0

    # Compose outputs
    if not df.empty:
        df["status"] = [
            _status_for(sd) for sd in df["settlement_date"].tolist()
        ]
        df["is_weekend_or_holiday"] = [
            _is_weekend_or_holiday(sd) for sd in df["settlement_date"].tolist()
        ]
        df["stp_ready_flag"] = stp_ready_series.values
        df["exposure_abs"] = exposure_abs.values if hasattr(exposure_abs, "values") else exposure_abs
    else:
        df["status"] = []
        df["is_weekend_or_holiday"] = []
        df["stp_ready_flag"] = []
        df["exposure_abs"] = []

    # KPIs
    try:
        counts_by_status = df["status"].value_counts(dropna=False).to_dict() if "status" in df.columns else {}
    except Exception:
        counts_by_status = {}
    try:
        exposure_by_status = (
            df.groupby(["status"], dropna=False)["exposure_abs"].sum().to_dict()
            if ("status" in df.columns and "exposure_abs" in df.columns)
            else {}
        )
    except Exception:
        exposure_by_status = {}
    total_exposure = float(df.get("exposure_abs", pd.Series(dtype=float)).sum()) if "exposure_abs" in df.columns else 0.0
    kpis = {
        "as_of": due_today,
        "counts_by_status": counts_by_status,
        "exposure_by_status": exposure_by_status,
        "total_exposure": total_exposure,
    }

    # Groupings
    try:
        by_counterparty = (
            df.groupby(["counterparty_legal_name", "status"], dropna=False)["exposure_abs"].sum().reset_index().sort_values("exposure_abs", ascending=False)
            if ("counterparty_legal_name" in df.columns)
            else pd.DataFrame()
        )
    except Exception:
        by_counterparty = pd.DataFrame()
    try:
        by_security = (
            df.groupby(["security_id", "status"], dropna=False)["exposure_abs"].sum().reset_index().sort_values("exposure_abs", ascending=False)
            if ("security_id" in df.columns)
            else pd.DataFrame()
        )
    except Exception:
        by_security = pd.DataFrame()

    return {"df_all": df, "kpis": kpis, "by_counterparty": by_counterparty, "by_security": by_security}

def page_overview():
    st.header("Investment Operations & Analytics")
    st.markdown(
        """
<div class="hero-subhead">Streamline operations with AI-powered trade capture, exception management, and settlement readiness.</div>
<div class="hero-subhead">Built for asset managers and hedge funds.</div>
""",
        unsafe_allow_html=True,
    )
    # Scoped CSS for Overview typography and lists
    st.markdown(
        """
<style>
.overview ul li { font-size: 1rem; line-height: 1.55; }
.overview ul li em, .overview ul li i { font-style: normal; }
.overview ul li strong { font-weight: 600; }
.hero-subhead {
  font-weight: 600;
  line-height: 1.45;
  color: inherit;
  font-size: 1.1rem;
  margin: -6px 0 12px 0;
}
@media (min-width: 768px) {
  .hero-subhead { font-size: 1.2rem; }
}
</style>
""",
        unsafe_allow_html=True,
    )

    # One-liner intro (scoped container)
    st.markdown(
        """
<div class="overview">
  <p>This app demonstrates a middle office workflow from trade capture through settlement preparation with KPIs built in. Upload an OMS trade capture file to generate outputs ready for STP, flag exceptions based on rules, and prepare for same day affirmation.
</div>
""",
        unsafe_allow_html=True,
    )

    # How it works
    st.subheader("How it works")
    st.markdown(
        """
<div class="overview">
  <ul>
    <li><strong>Trade Capture & Data Quality Review:</strong> upload the OMS CSV, map required fields, and run presence/type/domain/reference checks; writes <strong>STP-ready</strong> and <strong>Exceptions</strong> outputs only on <strong>new</strong> OMS uploads.</li>
    <li><strong>Reference Data:</strong> Security Master and Counterparty SSI lookups.</li>
    <li><strong>Enriched Positions & KPIs:</strong> portfolio stats (Market Value, <strong>Unrealized Gain/Loss</strong> & <strong>Unrealized Gain/Loss %</strong>, Day Change %), exceptions view, and settlement exposure by counterparty (unsettled).</li>
    <li><strong>Lifecycle Tracking:</strong> Pending / Settled / Failed derived from rules.</li>
    <li><strong>Holiday Awareness:</strong> U.S. weekend/holiday checks; see <strong>DTC Settlement Holiday Schedule</strong> for the 2025 table.</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    # Quick start
    st.subheader("Quick start")
    st.markdown(
        """
1) Go to **Trade Capture & Data Quality Review** and upload your OMS trade CSV (map fields if prompted).
2) Review exceptions and STP-ready outputs on completion (writes/updates only on new files).
3) Explore **Validation Rules**, **Reference Data**, **Trade Lifecycle**, and (optionally) **Post-Settlement**.
4) Run the **Ops Co-pilot (AI Assistant)** to turn your data into verified insights with cited sources using context aware retrieval (RAG) for faster decisions, fewer manual checks, and a lightweight audit trail.
"""
    )

    # Helpful actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Open Trade Capture & Data Quality Review →", use_container_width=True):
            st.session_state["__nav_next"] = "Trade Capture & Data Quality Review"
            st.rerun()
    with col2:
        if st.button("View Validation Rules →", use_container_width=True):
            st.session_state["__nav_next"] = "Validation Rules"
            st.rerun()

    

    # Glossary
    with st.expander("Quick glossary", expanded=False):
        st.markdown(
            """
- **T+1** — most U.S. securities settle one business day after trade date, so matching and affirmation must happen faster.
- **STP (Straight-Through Processing)** — trades flow from capture to settlement with no manual intervention.
- **Trade enrichment** — filling missing details (e.g., identifiers, currency, SSIs) from reference data after initial capture.
- **Security Master** — static reference data about instruments; **Client/Counterparty Master** — static data about clients and brokers.
- **SSI (Standing Settlement Instructions)** — default delivery/payment instructions that custodians use to settle trades.
- **Affirmation / Confirmation** — agreeing trade economics with the broker within tolerances before settlement.
- **Break** — any discrepancy (data, price, quantity, SSI) that must be resolved before settlement.
- **Exception management** — detecting, categorizing, and clearing breaks to maximize on-time settlement.
- **Lifecycle statuses** — Pending (not settled yet), Settled (on/after value date with no breaks), Failed (missed settlement due to open issues).
"""
        )

 

def page_trade_lifecycle():
    # NOTE: Single source of truth for Trade Lifecycle lives in app.py
    pass

def page_help_about():
    st.header("Help & About")
    st.write("Documentation, usage tips, and version information.")

# Override with OMS Trade Capture & Data Quality Review (pre‑CTM validations)
def page_upload_review_v2():
    import os
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import streamlit as st
    from datetime import date as _date, timedelta as _td, datetime as _dt
    from rule_catalog import load_rule_catalog, rule_meta, severity_badge_text

    st.title("Trade Capture & Data Quality Review")
    st.caption("Upload OMS trades, map required fields, and run pre-CTM checks against Security Master, counterparty SSIs, and the holiday calendar. Review Exceptions and STP-ready trades below.")


    # Initialize guarded write state
    if "oms_file_hash" not in st.session_state:
        st.session_state["oms_file_hash"] = None
    if "trades_clean_df" not in st.session_state:
        st.session_state["trades_clean_df"] = None
    if "trades_exceptions_df" not in st.session_state:
        st.session_state["trades_exceptions_df"] = None

    def _get_data_dir() -> Path:
        try:
            from app import get_data_dir as _gdd
            return _gdd()
        except Exception:
            p = Path(__file__).resolve().parent / "data"
            p.mkdir(parents=True, exist_ok=True)
            return p

    def _holiday_set() -> set[str]:
        try:
            from app import holiday_set as _hs
            return _hs
        except Exception:
            return set()

    def _weekend_or_holiday_flag(sd) -> tuple[bool, str | None]:
        try:
            from app import weekend_or_holiday_flag as _woh
            return _woh(sd, _holiday_set())
        except Exception:
            try:
                d = pd.to_datetime(sd, errors="coerce").date()
            except Exception:
                return False, None
            if d is None:
                return False, None
            if d.weekday() >= 5:
                return True, "Settlement date falls on a weekend/holiday"
            return False, None

    def add_business_days(d: _date, n: int) -> _date:
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        cur = d
        while remaining > 0:
            cur = cur + _td(days=step)
            if cur.weekday() < 5:
                remaining -= 1
        return cur

    data_dir = _get_data_dir()

    @st.cache_data(show_spinner=False)
    def _load_yaml_catalog() -> dict:
        try:
            return load_rule_catalog()
        except Exception:
            return {"by_code": {}, "list": []}

    @st.cache_data(show_spinner=False)
    def _load_security_master() -> pd.DataFrame:
        candidates = [
            data_dir / "security_master_brk_q2_2025.csv",
            data_dir / "security_master.csv",
        ]
        for p in candidates:
            try:
                if p.exists():
                    df = pd.read_csv(p)
                    # Normalize columns and required fields
                    df.columns = [str(c).strip() for c in df.columns]
                    cols_lower = {c.lower(): c for c in df.columns}
                    if "security_id" not in df.columns and "ticker" in df.columns:
                        df["security_id"] = (
                            df[cols_lower.get("ticker", "ticker")]
                                .astype(str).str.strip().str.upper()
                        )
                    if "security_id" in df.columns:
                        df["security_id"] = df["security_id"].astype(str).str.strip().str.upper()
                    if "settlement_cycle" not in df.columns:
                        df["settlement_cycle"] = "T+1"
                    if "asset_class" not in df.columns:
                        df["asset_class"] = "Equity"
                    if "currency" not in df.columns:
                        df["currency"] = "USD"
                    # Coerce active_flag to boolean (default True)
                    if "active_flag" in df.columns:
                        def _to_bool(x):
                            s = str(x).strip().lower()
                            return s in ("true", "1", "y", "yes", "t")
                        df["active_flag"] = df["active_flag"].apply(_to_bool)
                    else:
                        df["active_flag"] = True
                    return df
            except Exception:
                pass
        return pd.DataFrame()

    @st.cache_data(show_spinner=False)
    def _load_ssi_master() -> pd.DataFrame:
        candidates = [
            data_dir / "counterparty_ssi_brokers.csv",
            data_dir / "counterparty_ssi.csv",
        ]
        for p in candidates:
            try:
                if p.exists():
                    df = pd.read_csv(p)
                    df.columns = [str(c).strip() for c in df.columns]
                    # Coerce active_flag to boolean if present
                    if "active_flag" in df.columns:
                        def _to_bool(x):
                            s = str(x).strip().lower()
                            return s in ("true", "1", "y", "yes", "t")
                        df["active_flag"] = df["active_flag"].apply(_to_bool)
                    # Normalize counterparty name whitespace
                    if "counterparty_legal_name" in df.columns:
                        df["counterparty_legal_name"] = (
                            df["counterparty_legal_name"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
                        )
                    return df
            except Exception:
                pass
        return pd.DataFrame()

    st.subheader("Upload OMS trade capture")

    default_name = "oms_trade_capture_2025-08-14_v2.csv"
    src_mode = st.radio("Source", ["Select from data/", "Upload CSV"], index=0, horizontal=True)
    df_raw = st.session_state.get("v2_oms_df")
    if src_mode == "Select from data/":
        options = []
        try:
            options = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
        except Exception:
            options = []
        if default_name in options:
            options = [default_name] + [f for f in options if f != default_name]
        choice = st.selectbox("Choose a CSV in data/", options=["— Select —"] + options, index=0)
        if choice != "— Select —" and st.button("Load selected file", type="primary"):
            try:
                file_path = data_dir / choice
                df_raw = pd.read_csv(file_path)
                st.session_state["v2_oms_df"] = df_raw
                try:
                    st.session_state["v2_oms_bytes"] = Path(file_path).read_bytes()
                except Exception:
                    st.session_state["v2_oms_bytes"] = df_raw.to_csv(index=False).encode("utf-8")
                st.success(f"Loaded {choice} with {len(df_raw)} rows")
            except Exception as e:
                st.error(f"Failed to read {choice}: {e}")
    else:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None and st.button("Load uploaded file", type="primary"):
            try:
                df_raw = pd.read_csv(up)
                st.session_state["v2_oms_df"] = df_raw
                try:
                    st.session_state["v2_oms_bytes"] = up.getvalue()
                except Exception:
                    try:
                        up.seek(0)
                        st.session_state["v2_oms_bytes"] = up.read()
                    except Exception:
                        st.session_state["v2_oms_bytes"] = df_raw.to_csv(index=False).encode("utf-8")
                st.success(f"Loaded uploaded file with {len(df_raw)} rows")
            except Exception as e:
                st.error(f"Failed to read upload: {e}")

    if df_raw is None:
        # Fresh view or refresh: clear in-memory artifacts; do not load from disk
        st.session_state["trades_clean_df"] = None
        st.session_state["trades_exceptions_df"] = None
        st.session_state["artifacts_committed_at"] = None
        st.info("Awaiting upload. Expected columns: trade_date, settlement_date, side, quantity, security_id, price, counterparty_legal_name")
        return

    st.subheader("Map required fields")
    required = ["trade_date","settlement_date","side","quantity","security_id","price","counterparty_legal_name"]
    cols_lower = {c.lower(): c for c in df_raw.columns}
    synonyms = {
        "security_id": ["security_id","ticker","symbol","instrument","securityid"],
        "counterparty_legal_name": ["counterparty_legal_name","counterparty","broker","broker_name","counterparty name"],
        "quantity": ["quantity","qty","shares"],
        "price": ["price","unit_cost","exec_price","execution_price"],
        "side": ["side","buy/sell","trade_side"],
        "trade_date": ["trade_date","tradedate","trade_dt","td"],
        "settlement_date": ["settlement_date","settl_date","sd","settlementdate"],
    }
    auto_map = {}
    for req in required:
        mapped = None
        for alias in synonyms.get(req, [req]):
            if alias.lower() in cols_lower:
                mapped = cols_lower[alias.lower()]
                break
        auto_map[req] = mapped

    map_state_key = "v2_field_map"
    cur_map = st.session_state.get(map_state_key, {}) or auto_map.copy()
    missing_keys = [k for k, v in cur_map.items() if v is None]
    if missing_keys:
        st.warning("Some required fields are unmapped. Please map them below.")
        cols = st.columns(3)
        for idx, key in enumerate(missing_keys):
            with cols[idx % 3]:
                cur_map[key] = st.selectbox(f"Map → {key}", options=[None] + list(df_raw.columns), index=0, key=f"map_{key}__v2")
        st.session_state[map_state_key] = cur_map
    else:
        st.session_state[map_state_key] = cur_map
    if any(not cur_map.get(k) for k in required):
        st.stop()

    df = pd.DataFrame({k: df_raw[cur_map[k]] for k in required})
    df["security_id"] = df["security_id"].astype(str).str.strip().str.upper()
    df["counterparty_legal_name"] = df["counterparty_legal_name"].astype(str).str.strip().str.replace(r"\s+"," ", regex=True)
    df["side"] = df["side"].astype(str).str.strip().str.lower().map({"buy":"Buy","sell":"Sell"})
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce").dt.date

    # Compute content hash of uploaded bytes (or normalized CSV fallback)
    try:
        raw_bytes = st.session_state.get("v2_oms_bytes")
        if raw_bytes is None:
            raw_bytes = df_raw.to_csv(index=False).encode("utf-8")
        new_hash = hashlib.sha256(raw_bytes).hexdigest()
    except Exception:
        new_hash = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()

    is_new_upload = (new_hash != st.session_state.get("oms_file_hash"))

    if is_new_upload:
        # Run validations and writes only on new upload
        sec = _load_security_master()
        ssi = _load_ssi_master()
        catalog = _load_yaml_catalog()
        if sec.empty:
            st.info("Security Master not found in data/. Reference checks will mark securities as unknown.")
        if ssi.empty:
            st.info("Counterparty SSI not found in data/. Reference checks will mark counterparties as unknown or incomplete.")

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

        settle_cycle_by_sec = {}
        sec_class_by_sec = {}
        if "security_id" in sec.columns:
            for _, r in sec.iterrows():
                sid = str(r.get("security_id",""))
                if not sid:
                    continue
                settle_cycle_by_sec[str(sid).upper()] = str(r.get("settlement_cycle","") or "").upper().replace(" ", "")
                sec_class_by_sec[str(sid).upper()] = str(r.get("asset_class",""))

        valid_securities = set([str(x).upper() for x in sec.get("security_id", pd.Series(dtype=str)).tolist()])
        valid_cpys = set([str(x).strip() for x in ssi.get("counterparty_legal_name", pd.Series(dtype=str)).tolist()])
        ssi_complete_by_cpy = {}
        if not ssi.empty:
            dep_col = "depository_account" if "depository_account" in ssi.columns else None
            cash_col = "cash_account" if "cash_account" in ssi.columns else None
            for _, r in ssi.iterrows():
                c = str(r.get("counterparty_legal_name", "")).strip()
                dep_ok = bool(str(r.get(dep_col, "") or "").strip()) if dep_col else False
                cash_ok = bool(str(r.get(cash_col, "") or "").strip()) if cash_col else False
                ssi_complete_by_cpy[c] = (dep_ok and cash_ok)

        required_fields = required
        def _row_rule_check(row):
            for rf in required_fields:
                v = row.get(rf)
                if v in (None, "") or (pd.isna(v) if not isinstance(v, str) else False):
                    return ("presence_missing_field", "Missing required field", False, False)
                
            if not (isinstance(row["quantity"], (int, float)) and float(row["quantity"]) > 0):
                return ("quantity_nonpositive", "Quantity must be > 0", False, False)
            if not (isinstance(row["price"], (int, float)) and float(row["price"]) > 0):
                return ("price_nonpositive", "Price must be > 0", False, False)
            if row.get("side") not in {"Buy", "Sell"}:
                return ("invalid_side", "Side must be Buy or Sell", False, False)

            sec_id = str(row.get("security_id", "")).upper().strip()
            if sec_id not in valid_securities:
                return ("unknown_security", "Security not found or inactive", True, False)
            cpy = str(row.get("counterparty_legal_name", "")).strip()
            if cpy not in valid_cpys:
                return ("unknown_counterparty", "Counterparty not found or inactive", True, False)
            # Enforce SSI completeness only for equities where applicable
            sec_class = str(sec_class_by_sec.get(sec_id, "")).strip().lower()
            if sec_class in ("equity", "equities", "stock", "us equities", "us equity", ""):
                if not bool(ssi_complete_by_cpy.get(cpy, False)):
                    return ("incomplete_ssi", "Missing SSI depository/cash details for US Equities", True, False)

            cycle = (settle_cycle_by_sec.get(sec_id, "") or "T+1").upper().replace(" ", "")
            td = row.get("trade_date"); sd = row.get("settlement_date")
            if isinstance(td, _date) and isinstance(sd, _date):
                if cycle in {"T+1", "T+2"}:
                    n = 1 if cycle == "T+1" else 2
                    expected = add_business_days(td, n)
                    if sd != expected:
                        return ("bad_settlement_cycle", "Settlement date does not match configured cycle", True, False)

            bad_flag, _ = _weekend_or_holiday_flag(sd)
            if bad_flag:
                return ("weekend_or_holiday_settlement", "Settlement date falls on a weekend/holiday", True, False)
            return (None, None, True, True)

        results = df.apply(_row_rule_check, axis=1, result_type="expand")
        results.columns = ["Rule Code", "Exception Reason", "_pass_schema", "stp_ready"]
        out = pd.concat([df, results], axis=1)

        def _meta_val(code: str, key: str, default: str = "") -> str:
            try:
                meta = rule_meta(code) or {}
                val = meta.get(key)
                return str(val) if val not in (None, "") else default
            except Exception:
                return default

        out["Category"] = out["Rule Code"].apply(lambda c: _meta_val(c, "category", ""))
        out["Severity"] = out["Rule Code"].apply(lambda c: _meta_val(c, "severity", ""))
        out["Owner"] = out["Rule Code"].apply(lambda c: _meta_val(c, "owner", ""))
        out["Severity Label"] = out["Severity"].apply(severity_badge_text)
        def _prefer_yaml_reason(code: str, cur: str | None) -> str | None:
            try:
                msg = _meta_val(code, "message", "")
                return msg if msg else cur
            except Exception:
                return cur
        out["Exception Reason"] = [
            _prefer_yaml_reason(c, r) for c, r in zip(out["Rule Code"].tolist(), out["Exception Reason"].tolist())
        ]

        total = len(out)
        pct_schema = (out["_pass_schema"].sum() / total * 100.0) if total else 0.0
        pct_stp = (out["stp_ready"].sum() / total * 100.0) if total else 0.0
        # KPI tiles are rendered below in a unified strip

        clean = out[out["stp_ready"] == True].copy()
        exceptions = out[out["stp_ready"] != True].copy()

        # Guarded writes under hash-change only
        try:
            (data_dir / "trades_clean.csv").write_bytes(clean.to_csv(index=False).encode("utf-8"))
            (data_dir / "trades_exceptions.csv").write_bytes(exceptions.to_csv(index=False).encode("utf-8"))
            st.session_state["oms_file_hash"] = new_hash
            st.session_state["trades_clean_df"] = clean.copy()
            st.session_state["trades_exceptions_df"] = exceptions.copy()
            st.session_state["artifacts_committed_at"] = _dt.now().isoformat(timespec="seconds")
            st.success("Wrote outputs to data/: trades_clean.csv, trades_exceptions.csv")
        except Exception as e:
            st.error(f"Failed to manage outputs: {e}")
    else:
        # No new upload: do not overwrite; render last artifacts
        ts = st.session_state.get("artifacts_committed_at")
        st.info(f"No new upload detected. Showing last artifacts from {ts}." if ts else "No new upload detected. Showing last saved artifacts.")
        # On refresh or cache clear, start with empty in-memory artifacts; then backfill from disk if available
        st.session_state["trades_clean_df"] = None
        st.session_state["trades_exceptions_df"] = None
        clean = None
        exceptions = None
        if not isinstance(clean, pd.DataFrame) or clean is None:
            try:
                p = data_dir / "trades_clean.csv"
                if p.exists():
                    clean = pd.read_csv(p)
                    st.session_state["trades_clean_df"] = clean.copy()
            except Exception:
                clean = None
        if not isinstance(exceptions, pd.DataFrame) or exceptions is None:
            try:
                p = data_dir / "trades_exceptions.csv"
                if p.exists():
                    exceptions = pd.read_csv(p)
                    st.session_state["trades_exceptions_df"] = exceptions.copy()
            except Exception:
                exceptions = None

        # KPIs (computed from existing artifacts when present)
        try:
            total = ((len(clean) if isinstance(clean, pd.DataFrame) else 0) +
                     (len(exceptions) if isinstance(exceptions, pd.DataFrame) else 0))
            pct_schema = 0.0
            pct_stp = 0.0
            if total > 0:
                pass_schema = 0
                if isinstance(clean, pd.DataFrame) and "_pass_schema" in clean.columns:
                    pass_schema += int(clean["_pass_schema"].sum())
                if isinstance(exceptions, pd.DataFrame) and "_pass_schema" in exceptions.columns:
                    pass_schema += int(exceptions["_pass_schema"].sum())
                pct_schema = (pass_schema / total) * 100.0
                pct_stp = ((len(clean) if isinstance(clean, pd.DataFrame) else 0) / total) * 100.0
            # KPI tiles are rendered below in a unified strip
            
            pass
        except Exception:
            pass

    st.divider()

    # Right-aligned artifacts status caption (if available)
    _ts_et = st.session_state.get("artifacts_committed_at_et")
    if _ts_et:
        _c1, _c2, _c3, _c4 = st.columns([1, 1, 1, 1])
        with _c4:
            st.caption(f"Artifacts last updated {_ts_et} ET")

    # Resolve dataframes from session or earlier variables; fallback to disk if none
    trades_exceptions_df = st.session_state.get("trades_exceptions_df") if isinstance(st.session_state.get("trades_exceptions_df"), pd.DataFrame) else exceptions
    trades_clean_df = st.session_state.get("trades_clean_df") if isinstance(st.session_state.get("trades_clean_df"), pd.DataFrame) else clean
    if not isinstance(trades_exceptions_df, pd.DataFrame) and not isinstance(trades_clean_df, pd.DataFrame):
        try:
            p_exc = data_dir / "trades_exceptions.csv"
            if p_exc.exists():
                trades_exceptions_df = pd.read_csv(p_exc)
            p_clean = data_dir / "trades_clean.csv"
            if p_clean.exists():
                trades_clean_df = pd.read_csv(p_clean)
        except Exception:
            pass
    if not isinstance(trades_exceptions_df, pd.DataFrame) and not isinstance(trades_clean_df, pd.DataFrame):
        st.info("Upload an OMS trade file to view KPIs and results.")
        return

    # Optional: exceptions_by_rule_df for Schema Pass % inference
    exceptions_by_rule_df = st.session_state.get("exceptions_by_rule_df")
    if not isinstance(exceptions_by_rule_df, pd.DataFrame):
        try:
            p_ex = data_dir / "exceptions_by_rule.csv"
            if p_ex.exists():
                exceptions_by_rule_df = pd.read_csv(p_ex)
        except Exception:
            exceptions_by_rule_df = None

    # KPI strip (two rows of up to 4 tiles)
    metrics = []  # list of (label, value)
    try:
        records_ingested = None
        if isinstance(trades_clean_df, pd.DataFrame) and isinstance(trades_exceptions_df, pd.DataFrame):
            records_ingested = len(trades_clean_df) + len(trades_exceptions_df)
            metrics.append(("Records Ingested", f"{records_ingested:,}"))

        # Schema Pass %
        schema_pct = None
        if isinstance(trades_clean_df, pd.DataFrame) and "_pass_schema" in trades_clean_df.columns:
            pass_schema = int(trades_clean_df["_pass_schema"].sum())
            if isinstance(trades_exceptions_df, pd.DataFrame) and "_pass_schema" in trades_exceptions_df.columns:
                pass_schema += int(trades_exceptions_df["_pass_schema"].sum())
            total_rows = records_ingested if records_ingested is not None else None
            if total_rows not in (None, 0):
                schema_pct = (pass_schema / total_rows) * 100.0
        elif isinstance(exceptions_by_rule_df, pd.DataFrame) and isinstance(records_ingested, int) and records_ingested > 0:
            schema_failed_count = None
            for col in ["schema_failed", "is_schema_failure", "schema_failures"]:
                if col in exceptions_by_rule_df.columns:
                    try:
                        schema_failed_count = int(exceptions_by_rule_df[col].sum())
                        break
                    except Exception:
                        pass
            if isinstance(schema_failed_count, int):
                schema_pct = 100.0 * (1.0 - (schema_failed_count / records_ingested))
        if isinstance(schema_pct, (int, float)):
            metrics.append(("Schema Pass %", f"{schema_pct:.1f}%"))

        # STP-Ready %
        if isinstance(records_ingested, int) and records_ingested > 0 and isinstance(trades_clean_df, pd.DataFrame):
            metrics.append(("STP-Ready %", f"{(len(trades_clean_df) / records_ingested) * 100.0:.1f}%"))

        # Exceptions
        exc_count = 0
        if isinstance(trades_exceptions_df, pd.DataFrame):
            exc_count = len(trades_exceptions_df)
        metrics.append(("Exceptions", f"{exc_count:,}"))

        # Helper to count by rule code safely
        def _count_by_rule(code: str) -> int:
            if not isinstance(trades_exceptions_df, pd.DataFrame) or "Rule Code" not in getattr(trades_exceptions_df, "columns", []):
                return 0
            try:
                return int((trades_exceptions_df["Rule Code"].astype(str).str.lower() == code).sum())
            except Exception:
                return 0

        metrics.append(("Unknown Security", f"{_count_by_rule('unknown_security'):,}"))
        metrics.append(("Unknown Counterparty", f"{_count_by_rule('unknown_counterparty'):,}"))
        metrics.append(("Weekend/Holiday SD", f"{_count_by_rule('weekend_or_holiday_settlement'):,}"))
        metrics.append(("Bad Settlement Cycle", f"{_count_by_rule('bad_settlement_cycle'):,}"))
    except Exception:
        pass

    if metrics:
        row1 = metrics[:4]
        row2 = metrics[4:8]
        if row1:
            cols = st.columns(4)
            for idx, (label, value) in enumerate(row1):
                with cols[idx]:
                    st.metric(label, value)
        if row2:
            cols = st.columns(4)
            for idx, (label, value) in enumerate(row2):
                with cols[idx]:
                    st.metric(label, value)

    # Exceptions table (simplified)
    st.subheader("Exceptions")
    if isinstance(trades_exceptions_df, pd.DataFrame) and not trades_exceptions_df.empty:
        ex_disp = trades_exceptions_df.copy()
        # Merge Security Master description (read-only)
        try:
            sec_df = _load_security_master()
            if isinstance(sec_df, pd.DataFrame) and not sec_df.empty:
                # Pick best available description column
                desc_candidates = ["description", "company_name", "Security Description", "security_description", "name"]
                desc_col = next((c for c in desc_candidates if c in sec_df.columns), None)
                if desc_col is not None:
                    lut = sec_df[["security_id", desc_col]].rename(columns={desc_col: "description"}).drop_duplicates()
                    ex_disp = ex_disp.merge(lut, on="security_id", how="left")
        except Exception:
            pass
        if "Severity Label" not in ex_disp.columns:
            try:
                base = ex_disp["Severity"] if "Severity" in ex_disp.columns else ex_disp.get("Rule Code")
                if base is not None:
                    ex_disp["Severity Label"] = base.astype(str).apply(severity_badge_text)
            except Exception:
                pass
        # Severity-first sort (Critical > High > Medium > Low), then Exception Reason
        sev_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        try:
            ex_disp["__sev_rank"] = ex_disp["Severity"].astype(str).str.lower().map(lambda s: sev_rank.get(s, 9))
        except Exception:
            try:
                ex_disp["__sev_rank"] = ex_disp["Severity Label"].astype(str).str.lower().map(
                    lambda s: 0 if "critical" in s else (1 if "high" in s else (2 if "medium" in s else (3 if "low" in s else 9)))
                )
            except Exception:
                ex_disp["__sev_rank"] = 9
        ex_disp = ex_disp.sort_values(["__sev_rank", "Exception Reason"], kind="stable")

        visible_cols = [
            "Severity Label", "Category", "Exception Reason",
            "trade_date", "settlement_date", "side", "quantity", "security_id", "description", "price", "counterparty_legal_name",
        ]
        visible_cols = [c for c in visible_cols if c in ex_disp.columns]
        st.dataframe(ex_disp[visible_cols].head(1000), hide_index=True, use_container_width=True)
    else:
        st.success("No Exceptions detected.")

    # STP-ready table (collapsed expander)
    if isinstance(trades_clean_df, pd.DataFrame) and not trades_clean_df.empty:
        clean_disp = trades_clean_df.copy()
        # Merge Security Master description (read-only)
        try:
            sec_df = _load_security_master()
            if isinstance(sec_df, pd.DataFrame) and not sec_df.empty:
                desc_candidates = ["description", "company_name", "Security Description", "security_description", "name"]
                desc_col = next((c for c in desc_candidates if c in sec_df.columns), None)
                if desc_col is not None:
                    lut = sec_df[["security_id", desc_col]].rename(columns={desc_col: "description"}).drop_duplicates()
                    clean_disp = clean_disp.merge(lut, on="security_id", how="left")
        except Exception:
            pass
        display_cols = [c for c in [
            "trade_date","settlement_date","side","quantity","security_id","description","price","counterparty_legal_name"
        ] if c in clean_disp.columns]
        exp = st.expander("STP-ready trades", expanded=False)
        exp.dataframe(clean_disp[display_cols].head(1000), hide_index=True, use_container_width=True)
    else:
        st.info("No STP-ready rows.")