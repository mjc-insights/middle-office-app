import os
import sys
import time
import argparse
from datetime import datetime, timedelta

import requests
import pandas as pd
from dotenv import load_dotenv


# Using entitlement=delayed to pull 15-minute delayed US market data (premium 75 plan).
ENTITLEMENT_MODE = "delayed"


def fetch_daily_adjusted(symbol: str, api_key: str, session: requests.Session) -> dict | None:
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": api_key,
        "entitlement": ENTITLEMENT_MODE,
    }
    try:
        resp = session.get("https://www.alphavantage.co/query", params=params, timeout=20)
        if not resp.ok:
            return None
        return resp.json()
    except Exception:
        return None


def find_price_on_or_before(date_str: str, series_daily: dict) -> tuple[float | None, str | None]:
    # Try date, then step back up to 10 calendar days
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None, None
    for _ in range(11):
        key = dt.strftime("%Y-%m-%d")
        row = series_daily.get(key)
        if isinstance(row, dict):
            close_val = row.get("4. close") or row.get("5. adjusted close")
            if close_val not in (None, ""):
                try:
                    return float(close_val), key
                except Exception:
                    return None, key
        dt -= timedelta(days=1)
    return None, None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Update unit_costs in positions CSV from Alpha Vantage daily data.")
    parser.add_argument("--csv", required=True, help="Path to positions CSV to update in place")
    args = parser.parse_args(argv)

    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not api_key:
        print("ERROR: ALPHA_VANTAGE_KEY not set in environment/.env", file=sys.stderr)
        return 1

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path, dtype={"ticker": str})
    if "ticker" not in df.columns or "trade_date" not in df.columns:
        print("ERROR: CSV must include 'ticker' and 'trade_date' columns", file=sys.stderr)
        return 1

    # Normalize tickers and trade dates
    df["ticker_norm"] = df["ticker"].astype(str).str.strip().str.upper()
    # Ensure trade_date is yyyy-mm-dd strings
    df["trade_date_str"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    tickers = sorted(df["ticker_norm"].dropna().unique())
    session = requests.Session()

    # Rate limit window for safety (< 75/min)
    window_start = time.monotonic()
    calls = 0

    symbol_to_series: dict[str, dict] = {}
    for sym in tickers:
        # Skip if already fetched (should not happen given unique loop)
        # Fetch full daily adjusted to cover all trade dates
        try:
            now = time.monotonic()
            elapsed = now - window_start
            if elapsed >= 60:
                window_start = now
                calls = 0
            if calls >= 70:
                sleep_s = max(0.0, 60 - elapsed)
                print(f"Rate limit guard: sleeping {sleep_s:.0f}s…")
                time.sleep(sleep_s)
                window_start = time.monotonic()
                calls = 0

            data = fetch_daily_adjusted(sym, api_key, session) or {}
            calls += 1
            series = data.get("Time Series (Daily)", {}) if isinstance(data, dict) else {}
            symbol_to_series[sym] = series
            if not series:
                note = (data or {}).get("Note") or (data or {}).get("Information") or (data or {}).get("Error Message")
                if note:
                    print(f"Warning: {sym} daily data note/info: {str(note)[:200]}")
        except Exception as e:
            print(f"Warning: failed to fetch {sym}: {e}")

    # Update unit_costs row-by-row
    updated_rows = 0
    changes: list[str] = []
    for idx, row in df.iterrows():
        sym = row["ticker_norm"]
        trade_date_str = row["trade_date_str"]
        series = symbol_to_series.get(sym, {})
        if not series or not isinstance(series, dict):
            continue
        price, used_date = find_price_on_or_before(trade_date_str, series)
        if price is not None:
            original = row.get("unit_cost")
            try:
                original_f = float(original)
            except Exception:
                original_f = None
            if original_f is None or abs(original_f - price) > 1e-6:
                df.at[idx, "unit_cost"] = price
                updated_rows += 1
                changes.append(f"{sym}: {original} -> {price} (date {used_date})")

    # Backup original and write updated
    backup_path = csv_path.replace(".csv", ".backup.csv")
    if not os.path.exists(backup_path):
        # Save a backup only once
        import shutil
        shutil.copyfile(csv_path, backup_path)

    df.drop(columns=["ticker_norm", "trade_date_str"], inplace=True)
    df.to_csv(csv_path, index=False)

    print(f"Updated {updated_rows} rows in {csv_path}.")
    if changes:
        print("Changes:")
        for line in changes:
            print(" - ", line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


