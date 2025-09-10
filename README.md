# Investment Operations & Analytics

**Streamline operations with AI-powered trade capture, exception management, and settlement readiness.**
**Built for asset managers and hedge funds.**

## What this app does
A Streamlit application that demonstrates a realistic middle-office workflow from trade capture through pre-settlement. Upload an OMS export to produce STP-ready outputs, flag rule-based exceptions, track lifecycle status, and prepare for T+1 affirmation.

## Key capabilities
- **Trade Capture & Data Quality Review**: Upload the OMS CSV, map required fields, and run presence, type, domain, and reference checks. The app writes **STP-ready** and **Exceptions** outputs only when a new OMS file is uploaded.  
- **Reference Data**: Security Master and Counterparty SSI lookups used by validation and exposure views.  
- **Enriched Positions & KPIs**: Portfolio stats (Market Value, Unrealized Gain/Loss and percent, day change percent), Exceptions view, and unsettled exposure by counterparty and by security.  
- **Lifecycle Tracking**: Pending, Settled, and Failed derived from rule outcomes.  
- **Holiday Awareness**: U.S. weekend and holiday checks with a dedicated **DTC Settlement Holiday Schedule (2025)** page.  
- **Affirmation Readiness**: Countdown to the 9:00 pm ET target with an **On Track / At Risk / Breach** status chip.

## Pages
- **Overview & Help**: Intro, how it works, Quick Start, and shortcuts to core tabs.  
- **Upload & Review**: Internal positions file upload, light summaries, and moved settings for Break Thresholds and Day-over-Day comparisons.  
- **Trade Capture & Data Quality Review**: OMS upload, validations, and STP-ready vs Exceptions outputs.  
- **Validation Rules**: Catalog view driven by `validation_rules.yaml` with helper functions and status utilities.  
- **Reference Data**: Security Master and Counterparty SSI lookups to support validation and exposure views.  
- **Trade Lifecycle**: Status and exposure summaries by counterparty and by security.  
- **Post-Settlement**: Placeholder for future reconciliation to custodian records.  
- **DTC Settlement Holiday Schedule**: Clean 2025 schedule table.  
- **AI Assistant**: "Ops Co-pilot" chat that can use repo context to answer questions and surface metrics.

## Data inputs and outputs
**Inputs**
- OMS trade capture CSV (required)  
- Reference CSVs: `security_master` and `counterparty_ssi_brokers`

**Derived outputs**
- **STP-ready** trades and **Exceptions** table  
- Portfolio KPIs and unsettled exposure (grouped by counterparty and by security)  
- Day-over-day change metrics when market data enrichment is enabled

## Market-data enrichment
Optional enrichment via **Alpha Vantage**:
- Uses regular trading hours intraday bars during market hours.  
- After 4:00 pm ET, it first attempts the official daily close. If not yet published, it falls back to the 16:00 intraday bar.  
- Includes simple retry, throttling, and caching.  
> Set `ALPHA_VANTAGE_KEY` in your environment to enable enrichment.

## AI Assistant (Ops Co-pilot)
- Repo-aware context so answers can cite or reason from project files when appropriate.  
- Lightweight intent router with generic tools and a data-dictionary builder for tables in session.  
- Bring-your-own OpenAI key in the sidebar field.  
> Set `OPENAI_API_KEY` in your environment to enable chat.

## Quick Start

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


2) Configure environment
Create a .env file in the project root (or export in your shell):
OPENAI_API_KEY=sk-...
ALPHA_VANTAGE_KEY=your_alpha_vantage_key

3) Run the app
streamlit run app.py


4) Use the workflow
⦁	Open Upload & Review to load a sample or internal positions file if needed.
⦁	Go to Trade Capture & Data Quality Review and upload your OMS CSV.
⦁	Review Exceptions and STP-ready outputs.
⦁	Explore Validation Rules, Reference Data, Trade Lifecycle, and DTC Settlement Holiday Schedule as needed.
⦁	Run AI Assistant to ask questions grounded in your data.

Repo highlights
⦁	app.py — Navigation, BYO OpenAI key field in the sidebar, holiday schedule view, affirmation countdown and status chip, Upload & Review page, and day-over-day helpers.
⦁	ui_pages.py — Overview copy, hero subheads (“Streamline operations…” and “Built for asset managers and hedge funds”), and UI helpers for KPIs and exposure groupings.
⦁	ai_assistant.py — Repo-aware RAG index (FAISS), tiny router, generic tools, data-dictionary builder, streaming status, and OpenAI client.
⦁	alpha_vantage.py — RTH-aware price logic and after-close daily-close fallback with caching and retry.
⦁	validation_rules.yaml — Rule catalog that drives the Validation Rules page.
⦁	Sample data: security_master, counterparty_ssi_brokers, oms_trade_capture

Roadmap
⦁	Build out Post-Settlement reconciliation to compare internal EOD positions to custodian records and confirm cash movements.
⦁	Expand rule catalog coverage and add more market locales.
⦁	Add broker confirm matching and automated affirmation metrics.

👤 Author Miller C.

This project is for demo purposes only.