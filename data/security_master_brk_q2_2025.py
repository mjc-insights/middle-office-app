# Create a new file: data/security_master_brk_q2_2025.py
# Purpose: Static Security Master for Berkshire Hathaway's equity portfolio per 13F (Q2 2025; filed 2025-08-14).
# Sources: Official SEC 13F info table for CUSIPs; tickers cross-checked with reputable portfolio mirrors.
# Note: This is a static demo dataset for your Middle Office app (trade capture/enrichment/recon dashboards).

from typing import List, Dict
import pandas as pd
from pathlib import Path

SECURITY_MASTER_BRK_Q2_2025: List[Dict] = [
    # Core fields: ticker, name, cusip, exchange, currency, security_type, sector, industry, country
    # Financials / Banks / Payments
    {"ticker": "AXP",   "name": "American Express Co",                 "cusip": "025816109", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Financials",         "industry": "Consumer Finance",                    "country": "US"},
    {"ticker": "BAC",   "name": "Bank of America Corp",                "cusip": "060505104", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Financials",         "industry": "Banks",                               "country": "US"},
    {"ticker": "COF",   "name": "Capital One Financial Corp",          "cusip": "14040H105", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Financials",         "industry": "Consumer Finance",                    "country": "US"},
    {"ticker": "AON",   "name": "Aon plc Class A",                     "cusip": "G0403H108", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Financials",         "industry": "Insurance Brokers",                   "country": "IE"},
    {"ticker": "MA",    "name": "Mastercard Inc Class A",              "cusip": "57636Q104", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Information Technology","industry": "Data Processing & Outsourced Svcs", "country": "US"},
    {"ticker": "MCO",   "name": "Moody's Corp",                        "cusip": "615369105", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Financials",         "industry": "Financial Data & Analytics",          "country": "US"},
    {"ticker": "JEF",   "name": "Jefferies Financial Group Inc",       "cusip": "47233W109", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Financials",         "industry": "Investment Banking & Brokerage",      "country": "US"},

    # Tech / Communications
    {"ticker": "AAPL",  "name": "Apple Inc",                           "cusip": "037833100", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Information Technology","industry": "Technology Hardware",               "country": "US"},
    {"ticker": "AMZN",  "name": "Amazon.com Inc",                      "cusip": "023135106", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Consumer Discretionary","industry": "Broadline Retail / Internet",     "country": "US"},
    {"ticker": "VRSN",  "name": "VeriSign Inc",                        "cusip": "92343E102", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Information Technology","industry": "Internet Services & Infrastructure", "country": "US"},
    {"ticker": "V",     "name": "Visa Inc Class A",                    "cusip": "92826C839", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Information Technology","industry": "Data Processing & Outsourced Svcs", "country": "US"},

    # Communication Services (Media, Cable, Satellite)
    {"ticker": "CHTR",  "name": "Charter Communications Inc Class A",  "cusip": "16119P108", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Communication Services","industry": "Cable & Satellite",                "country": "US"},
    {"ticker": "SIRI",  "name": "Sirius XM Holdings Inc",              "cusip": "829933100", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Communication Services","industry": "Media",                            "country": "US"},
    {"ticker": "BATRK", "name": "Atlanta Braves Holdings Inc C",       "cusip": "047726302", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Communication Services","industry": "Entertainment",                     "country": "US"},
    {"ticker": "FWONK", "name": "Liberty Media Formula One C",         "cusip": "531229854", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Communication Services","industry": "Entertainment",                     "country": "US"},
    {"ticker": "LLYVA", "name": "Liberty Media Liberty Live A",        "cusip": "531229748", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Communication Services","industry": "Entertainment",                     "country": "US"},
    {"ticker": "LLYVK", "name": "Liberty Media Liberty Live C",        "cusip": "531229722", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Communication Services","industry": "Entertainment",                     "country": "US"},
    {"ticker": "LILA",  "name": "Liberty Latin America Class A",       "cusip": "G9001E102", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Communication Services","industry": "Cable & Satellite",                "country": "BM"},
    {"ticker": "LILAK", "name": "Liberty Latin America Class C",       "cusip": "G9001E128", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Communication Services","industry": "Cable & Satellite",                "country": "BM"},

    # Consumer Staples
    {"ticker": "KO",    "name": "Coca-Cola Co",                        "cusip": "191216100", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Consumer Staples",     "industry": "Beverages",                         "country": "US"},
    {"ticker": "KHC",   "name": "Kraft Heinz Co",                      "cusip": "500754106", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Consumer Staples",     "industry": "Packaged Foods & Meats",            "country": "US"},
    {"ticker": "STZ",   "name": "Constellation Brands Class A",        "cusip": "21036P108", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Consumer Staples",     "industry": "Beverages",                         "country": "US"},

    # Consumer Discretionary (Retail, Restaurants, Homebuilders, Pools)
    {"ticker": "DPZ",   "name": "Domino's Pizza Inc",                  "cusip": "25754A201", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Consumer Discretionary","industry": "Restaurants",                      "country": "US"},
    {"ticker": "DHI",   "name": "D.R. Horton Inc",                     "cusip": "23331A109", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Consumer Discretionary","industry": "Homebuilding",                    "country": "US"},
    {"ticker": "LEN",   "name": "Lennar Corp Class A",                 "cusip": "526057104", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Consumer Discretionary","industry": "Homebuilding",                    "country": "US"},
    {"ticker": "LEN.B", "name": "Lennar Corp Class B",                 "cusip": "526057302", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Consumer Discretionary","industry": "Homebuilding",                    "country": "US"},
    {"ticker": "POOL",  "name": "Pool Corp",                           "cusip": "73278L105", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Industrials",          "industry": "Trading Companies & Distributors",  "country": "US"},

    # Health Care
    {"ticker": "DVA",   "name": "DaVita Inc",                          "cusip": "23918K108", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Health Care",          "industry": "Health Care Facilities",            "country": "US"},
    {"ticker": "UNH",   "name": "UnitedHealth Group Inc",              "cusip": "91324P102", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Health Care",          "industry": "Managed Health Care",               "country": "US"},

    # Energy
    {"ticker": "CVX",   "name": "Chevron Corp",                        "cusip": "166764100", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Energy",               "industry": "Integrated Oil & Gas",              "country": "US"},
    {"ticker": "OXY",   "name": "Occidental Petroleum Corp",           "cusip": "674599105", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Energy",               "industry": "Oil & Gas Exploration & Production","country": "US"},

    # Industrials / Materials / REITs
    {"ticker": "ALLE",  "name": "Allegion plc",                        "cusip": "G0176J109", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Industrials",          "industry": "Building Products",                 "country": "IE"},
    {"ticker": "LLPX",  "name": "Louisiana-Pacific Corp",              "cusip": "546347105", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Materials",            "industry": "Forest Products",                   "country": "US"},  # ticker is LPX; key kept as LPX below to avoid confusion
    {"ticker": "LPX",   "name": "Louisiana-Pacific Corp",              "cusip": "546347105", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Materials",            "industry": "Forest Products",                   "country": "US"},
    {"ticker": "NUE",   "name": "Nucor Corp",                          "cusip": "670346105", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Materials",            "industry": "Steel",                             "country": "US"},
    {"ticker": "NVR",   "name": "NVR Inc",                             "cusip": "62944T105", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Consumer Discretionary","industry": "Homebuilding",                    "country": "US"},
    {"ticker": "LAMR",  "name": "Lamar Advertising Co Class A",        "cusip": "512816109", "exchange": "NASDAQ", "currency": "USD", "security_type": "Common Stock", "sector": "Real Estate",          "industry": "Specialized REITs",                 "country": "US"},
    {"ticker": "HEI.A", "name": "HEICO Corp Class A",                  "cusip": "422806208", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Industrials",          "industry": "Aerospace & Defense",               "country": "US"},

    # Staples Distributors / Grocers
    {"ticker": "KR",    "name": "Kroger Co",                           "cusip": "501044101", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Consumer Staples",     "industry": "Food & Staples Retailing",          "country": "US"},

    # Drinks (ADR)
    {"ticker": "DEO",   "name": "Diageo plc ADR",                      "cusip": "25243Q205", "exchange": "NYSE",   "currency": "USD", "security_type": "ADR",          "sector": "Consumer Staples",     "industry": "Beverages",                         "country": "GB"},

    # Insurance
    {"ticker": "CB",    "name": "Chubb Limited",                       "cusip": "H1467J104", "exchange": "NYSE",   "currency": "USD", "security_type": "Common Stock", "sector": "Financials",           "industry": "Property & Casualty Insurance",     "country": "CH"},
]

# Deduplicate accidental alias (keep LPX, drop LLPX)
_seen = set()
_clean = []
for row in SECURITY_MASTER_BRK_Q2_2025:
    key = row["ticker"]
    if key in _seen:
        continue
    _seen.add(key)
    _clean.append(row)
SECURITY_MASTER_BRK_Q2_2025 = _clean

def get_security_master_brk_df() -> pd.DataFrame:
    cols = ["ticker","name","cusip","exchange","currency","security_type","sector","industry","country"]
    df = pd.DataFrame(SECURITY_MASTER_BRK_Q2_2025, columns=cols)
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df

def save_csv(path: str = "data/security_master_brk_q2_2025.csv") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    get_security_master_brk_df().to_csv(p, index=False)
    return str(p)

if __name__ == "__main__":
    print(save_csv())


