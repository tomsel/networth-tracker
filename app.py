import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import gspread
import requests
from google.oauth2.service_account import Credentials
from helper_util import (
    read_google_sheet_to_df,
    load_and_normalize_main,
    load_and_normalize_static,
    fill_missing_amounts,
    add_amount_display,
    combine_snapshot,
    month_add,
    fmt_money,
)

@st.cache_data(ttl=3600)
def fetch_usd_sek_rate() -> float | None:
    try:
        resp = requests.get(
            "https://api.frankfurter.app/latest?from=USD&to=SEK",
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        return float(payload["rates"]["SEK"])
    except Exception:
        return None

st.set_page_config(page_title="Net Worth Tracker", layout="wide")
page = st.sidebar.radio(
    "Menu",
    ["Net Worth", "Instructions"],
    index=0
)

if page == "Net Worth":


    # Canonical schema
    BASE_COLS_MAIN = ["date", "owner", "category", "amount_sek", "amount_usd"]
    BASE_COLS_STATIC = ["owner", "category", "amount_sek", "amount_usd"]

    VALID_CATEGORIES = {"Cash", "Investments", "Debt", "Real Estate", "Pension"}

    # -----------------------------
    # UI
    # -----------------------------
    st.title("Net Worth — Tomas & Carlyn")

    with st.sidebar:
        st.header("Data & Scenario")

        data_source = st.radio("Data source", ["Google Sheet","Upload CSV"], index=0)

        uploaded_main = None
        uploaded_static = None

        sheet_url = "https://docs.google.com/spreadsheets/d/1xgeIBBnYeseXoIhXzxPd7GcCE7jszRkpGkwx_VS6NRw/edit?gid=0#gid=0"
        main_ws = "Main"
        static_ws = "Static Asset"
        sa_path = "secrets/gcp_sa.json"

        if data_source == "Upload CSV":
            uploaded_main = st.file_uploader("Upload MAIN CSV (monthly snapshots)", type=["csv"])
            uploaded_static = st.file_uploader("Upload STATIC CSV (optional)", type=["csv"])
            st.caption("Static CSV is for things like Real Estate + Debt that you don't want to re-enter every month.")

        display_currency = st.radio("Display currency", ["SEK", "USD"], index=0)

        fx_live = fetch_usd_sek_rate()
        if fx_live is None:
            st.warning("Live FX unavailable. Please enter a manual FX rate.")
            fx = st.number_input(
                "FX rate (SEK per 1 USD)",
                min_value=0.1,
                value=10.5,
                step=0.1,
                help="Fallback used only when live FX is unavailable.",
            )
        else:
            fx = fx_live
            st.caption(f"Live FX: {fx:.4f} SEK per 1 USD")

        st.divider()
        st.subheader("Projection assumptions")

        annual_return_investments = st.number_input(
            "Investments annual return (%)",
            min_value=-50.0,
            max_value=50.0,
            value=7.0,
            step=0.5,
        ) / 100.0

        annual_return_cash = st.number_input(
            "Cash annual return (%)",
            min_value=-50.0,
            max_value=50.0,
            value=2.0,
            step=0.5,
        ) / 100.0

        annual_return_pension = st.number_input(
            "Pension annual return (%)",
            min_value=-50.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="Only used if you include pension in net worth.",
        ) / 100.0

        debt_amort_annual = st.number_input(
            "Debt amortization per year (%)",
            min_value=0.0,
            max_value=20.0,
            value=1.0,
            step=0.1,
            help="1% per year => ~0.083% per month (no interest modeled).",
        ) / 100.0

        monthly_savings = st.number_input(
            "Monthly savings (added to Investments)",
            min_value=0.0,
            value=35000.0 if display_currency == "SEK" else 35000.0 / fx,
            step=1000.0 if display_currency == "SEK" else 100.0,
        )

        months_forward = st.slider("Months forward", 6, 240, 120)

        st.divider()
        include_pension_in_networth = st.checkbox("Include Pension in Net Worth", value=False)  # default off

        goal_value = st.number_input(
            "Goal value",
            min_value=0.0,
            value=10_000_000.0 if display_currency == "SEK" else 10_000_000.0 / fx,
            step=100_000.0 if display_currency == "SEK" else 10_000.0,
        )

    # -----------------------------
    # Load data (Main + Static)
    # -----------------------------
    df_main_raw = None
    df_static_raw = None

    if data_source == "Upload CSV":
        if not uploaded_main:
            st.info("Upload your MAIN CSV to begin.")
            st.stop()
        df_main_raw = pd.read_csv(uploaded_main)

        if uploaded_static:
            df_static_raw = pd.read_csv(uploaded_static)
        else:
            df_static_raw = pd.DataFrame(columns=BASE_COLS_STATIC)

    else:
        if not sheet_url.strip():
            st.info("Paste the Google Sheet URL to begin.")
            st.stop()

        # Main is required
        df_main_raw = read_google_sheet_to_df(sheet_url, main_ws, sa_path.strip() or None)

        # Static is optional (if tab missing or empty -> empty df)
        try:
            df_static_raw = read_google_sheet_to_df(sheet_url, static_ws, sa_path.strip() or None)
        except Exception:
            df_static_raw = pd.DataFrame(columns=BASE_COLS_STATIC)

        if df_main_raw is None or df_main_raw.empty:
            st.error("Could not read any rows from the MAIN sheet. Check sharing + worksheet name.")
            st.stop()

    try:
        df_main = load_and_normalize_main(df_main_raw)
        df_static = load_and_normalize_static(df_static_raw)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Auto-fill missing SEK/USD using the FX rate
    df_main = fill_missing_amounts(df_main, fx)
    df_static = fill_missing_amounts(df_static, fx)

    # Add display values
    df_main = add_amount_display(df_main, display_currency, fx)
    df_static = add_amount_display(df_static, display_currency, fx)

    # -----------------------------
    # Latest snapshot (Main latest date + Static overlay)
    # -----------------------------
    latest_date = df_main["date"].max()
    main_snap = df_main[df_main["date"] == latest_date].copy()
    snap = combine_snapshot(main_snap, df_static)

    # Debt negative in net worth
    snap["signed_amount"] = snap["amount_display"]
    snap.loc[snap["category"] == "Debt", "signed_amount"] = -snap.loc[snap["category"] == "Debt", "amount_display"]

    def is_in_networth(row):
        return not (row["category"] == "Pension" and not include_pension_in_networth)

    snap["in_networth"] = snap.apply(is_in_networth, axis=1)
    current_networth = float(snap.loc[snap["in_networth"], "signed_amount"].sum())

    st.subheader(f"Latest snapshot: {latest_date}")
    c1, c2 = st.columns(2)
    c1.metric(f"Net Worth ({display_currency})", fmt_money(current_networth))
    c2.metric("Rows in snapshot", str(len(snap)))

    show_cols = ["owner", "category", "amount_display"]
    tbl = snap[show_cols].rename(columns={"amount_display": f"amount_{display_currency.lower()}"})
    st.dataframe(tbl.sort_values(["owner", "category"]), use_container_width=True, hide_index=True)

    # -----------------------------
    # Projection model (from latest snapshot)
    # -----------------------------
    def start_value(cat):
        v = snap.loc[snap["category"] == cat, "amount_display"].sum()
        return float(v) if pd.notna(v) else 0.0

    start_cash = start_value("Cash")
    start_inv = start_value("Investments")
    start_debt = start_value("Debt")          # stored positive; subtract later
    start_re = start_value("Real Estate")
    start_pension = start_value("Pension")

    monthly_r_inv = annual_return_investments / 12.0
    monthly_r_cash = annual_return_cash / 12.0
    monthly_r_pension = annual_return_pension / 12.0
    debt_amort_monthly = debt_amort_annual / 12.0

    months = np.arange(0, months_forward + 1)

    inv = np.zeros_like(months, dtype=float)
    cash = np.zeros_like(months, dtype=float)
    debt = np.zeros_like(months, dtype=float)
    re = np.zeros_like(months, dtype=float)
    pension = np.zeros_like(months, dtype=float)

    inv[0] = start_inv
    cash[0] = start_cash
    debt[0] = start_debt
    re[0] = start_re
    pension[0] = start_pension

    for i in range(1, len(months)):
        cash[i] = cash[i - 1] * (1 + monthly_r_cash)
        inv[i] = inv[i - 1] * (1 + monthly_r_inv) + monthly_savings
        re[i] = re[i - 1]  # flat
        pension[i] = pension[i - 1] * (1 + monthly_r_pension)
        debt[i] = debt[i - 1] * (1 - debt_amort_monthly)

    networth_forecast = cash + inv + re - debt + (pension if include_pension_in_networth else 0.0)
    date_axis = [month_add(latest_date, int(m)) for m in months]

    proj = pd.DataFrame({
        "date": date_axis,
        "month": months,
        "Net Worth": networth_forecast,
        "Cash": cash,
        "Investments": inv,
        "Real Estate": re,
        "Debt": -debt,  # show negative
        "Pension": pension,
    })

    # -----------------------------
    # Plot: Utveckling + Prognos + Sparplan
    # -----------------------------
    st.divider()
    st.subheader("Projection")

    def compute_snapshot_networth_for_date(d) -> float:
        main_snap_d = df_main[df_main["date"] == d].copy()
        snap_d = combine_snapshot(main_snap_d, df_static)

        snap_d["signed_amount"] = snap_d["amount_display"]
        snap_d.loc[snap_d["category"] == "Debt", "signed_amount"] = -snap_d.loc[snap_d["category"] == "Debt", "amount_display"]

        snap_d["in_networth"] = snap_d.apply(is_in_networth, axis=1)
        return float(snap_d.loc[snap_d["in_networth"], "signed_amount"].sum())

    # Actual series (Utveckling) from Main dates
    actual_rows = []
    for d in sorted(df_main["date"].dropna().unique()):
        nw = compute_snapshot_networth_for_date(d)
        actual_rows.append({"date": pd.to_datetime(d), "value": nw, "series": "Utveckling"})
    actual_df = pd.DataFrame(actual_rows)

    # Forecast series (Prognos) from latest snapshot onward
    forecast_df = proj.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    forecast_df = forecast_df[["date", "Net Worth"]].rename(columns={"Net Worth": "value"})
    forecast_df["series"] = "Prognos"

    # Sparplan: straight line from earliest snapshot, no return, just monthly savings
    first_date = actual_df["date"].min().date()
    first_nw = compute_snapshot_networth_for_date(first_date)

    plan_dates = [month_add(first_date, int(m)) for m in months]
    plan_df = pd.DataFrame({
        "date": pd.to_datetime(plan_dates),
        "value": first_nw + (monthly_savings * months),
        "series": "Sparplan",
    })

    plot_df = pd.concat([actual_df, forecast_df, plan_df], ignore_index=True)

    # --- Plotly chart (better hover) ---
    plot_df_plotly = plot_df.copy()
    plot_df_plotly["date"] = pd.to_datetime(plot_df_plotly["date"])

    fig = px.line(
        plot_df_plotly,
        x="date",
        y="value",
        color="series",
        line_dash="series",
        category_orders={"series": ["Utveckling", "Prognos", "Sparplan"]},
    )

    # Nice hover: one tooltip box for all series at that date
    fig.update_traces(mode="lines")
    fig.update_layout(
        hovermode="x unified",
        height=420,
        legend_title_text="",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    # Make y-axis readable
    fig.update_yaxes(tickformat=",.0f")

    # Hide the annoying toolbar
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Goal ETA based on forecast net worth
    hit_idx = np.where(networth_forecast >= goal_value)[0]
    if len(hit_idx) > 0:
        m_hit = int(hit_idx[0])
        d_hit = date_axis[m_hit]
        st.success(f"Goal reached around: {d_hit} (month {m_hit})")
    else:
        st.info("Goal not reached within the selected horizon.")

    with st.expander("Show projection table"):
        st.dataframe(
            proj[["date", "month", "Net Worth", "Cash", "Investments", "Real Estate", "Debt", "Pension"]],
            use_container_width=True,
            hide_index=True,
        )

if page == "Instructions":
    st.title("How to use the Net Worth Tracker")

    st.markdown("""
### Google Sheet (click to edit)
👉 **[Open our Net Worth Google Sheet](https://docs.google.com/spreadsheets/d/1xgeIBBnYeseXoIhXzxPd7GcCE7jszRkpGkwx_VS6NRw/edit?gid=0#gid=0)**

(Requires Google access)

### Overview
This app automatically reads our Google Sheet and shows:
- Current net worth
- Historical development
- Forecast and savings plan

You normally **do not need to change anything in the app**.
All updates are done in Google Sheets.

---

## Sheet structure

The Google Sheet has **two tabs**:

### 1. Main (monthly updates)
This is the only sheet you update regularly.

**Each row = one snapshot for one person & category.**

Required columns:

- **date** → The snapshot date (example: `2025-03-01`)
- **owner** → `Tomas` or `Carlyn`
- **category** → Must be ONE of:
  - Cash
  - Investments
  - Pension
- **amount_sek** → Amount in SEK (numbers only)
- **amount_usd** → Amount in USD (optional if you filled SEK)

👉 You normally update this **once per month**.

---

### 2. Static Asset (rarely changed)
This sheet is for things that do **not change every month**, such as:
- Real Estate
- Debt (mortgage)

Required columns:

- **owner**
- **category** → Must be:
  - Real Estate
  - Debt
- **amount_sek**
- **amount_usd** (optional if you filled SEK)

👉 Only update this sheet if:
- We buy/sell property
- Mortgage changes significantly

---

## Important rules (very important)

- ❌ Do NOT change column names
- ❌ Do NOT add new categories
- ❌ Do NOT delete existing tabs
- ✅ Only add new rows

If something looks wrong in the app, it is almost always caused by:
- Misspelled category
- Date not filled in
- Text instead of numbers in amount columns

---

## What you never need to touch

You do NOT need to:
- Log in to Streamlit
- Paste any URLs
- Upload files
- Change settings

Just update the Google Sheet ❤️

---

If something breaks: tell Tomas 😄
""")
