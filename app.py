import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Net Worth Tracker", layout="wide")

# Canonical schema (no include_in_goal anymore)
BASE_COLS = ["date", "owner", "category", "amount_sek", "amount_usd"]

VALID_CATEGORIES = {"Cash", "Investments", "Debt", "Real Estate", "Pension"}

# -----------------------------
# Helpers
# -----------------------------
def parse_date_col(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def safe_float_series(s: pd.Series) -> pd.Series:
    # Handles: "35 000,00", "35 000,00", "", etc.
    return (
        s.astype(str)
        .str.replace("\u00a0", " ", regex=False)   # NBSP -> space
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "None": np.nan, "nan": np.nan})
        .astype(float)
    )

def month_add(d, m):
    return (pd.Period(d, freq="M") + m).to_timestamp().date()

def fmt_money(x):
    return f"{x:,.0f}".replace(",", " ")

def canonical_category(x: str) -> str:
    s = str(x).strip()
    mapping = {
        "Investment": "Investments",
        "Investments": "Investments",
        "Real estate": "Real Estate",
        "Real Estate": "Real Estate",
        "Cash": "Cash",
        "Debt": "Debt",
        "Pension": "Pension",
    }
    return mapping.get(s, s)

def read_google_sheet_to_df(sheet_url: str, worksheet: str, sa_json_path: str | None):
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    if sa_json_path:
        creds = Credentials.from_service_account_file(sa_json_path, scopes=scopes)
    else:
        creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=scopes)

    gc = gspread.authorize(creds)
    sh = gc.open_by_url(sheet_url)
    ws = sh.worksheet(worksheet)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()

    header = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=header)

def load_and_normalize(df_raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in BASE_COLS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Expected: {BASE_COLS}")

    df = df_raw.copy()
    df["date"] = parse_date_col(df["date"])
    df = df.dropna(subset=["date"])

    df["owner"] = df["owner"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).apply(canonical_category)

    df["amount_sek"] = safe_float_series(df["amount_sek"])
    df["amount_usd"] = safe_float_series(df["amount_usd"])

    unknown_cats = sorted(set(df["category"].unique()) - VALID_CATEGORIES)
    if unknown_cats:
        st.warning(f"Unknown category values found: {unknown_cats}\nAllowed: {sorted(VALID_CATEGORIES)}")

    return df

# -----------------------------
# UI
# -----------------------------
st.title("Net Worth — Tomas & Carlyn")

with st.sidebar:
    st.header("Data & Scenario")

    data_source = st.radio("Data source", ["Upload CSV", "Google Sheet"], index=0)

    uploaded = None
    sheet_url = ""
    worksheet_name = "Main"
    sa_path = "secrets/gcp_sa.json"

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
    else:
        sheet_url = st.text_input("Google Sheet URL", value="")
        worksheet_name = st.text_input("Worksheet name (tab)", value="Main")
        st.caption("Local: put your service account key at secrets/gcp_sa.json")
        sa_path = st.text_input("Service account JSON path (local)", value="secrets/gcp_sa.json")

    display_currency = st.radio("Display currency", ["SEK", "USD"], index=0)

    fx = st.number_input(
        "FX rate (SEK per 1 USD)",
        min_value=0.1,
        value=10.5,
        step=0.1,
        help="Used only for display conversion if one side is missing.",
    )

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
        value=0.0,
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
# Load data
# -----------------------------
df_raw = None

if data_source == "Upload CSV":
    if not uploaded:
        st.info("Upload your CSV to begin.")
        st.stop()
    df_raw = pd.read_csv(uploaded)
else:
    if not sheet_url.strip():
        st.info("Paste the Google Sheet URL to begin.")
        st.stop()

    try:
        df_raw = read_google_sheet_to_df(sheet_url, worksheet_name, sa_path.strip() or None)
    except FileNotFoundError:
        df_raw = read_google_sheet_to_df(sheet_url, worksheet_name, None)

    if df_raw is None or df_raw.empty:
        st.error("Could not read any rows from the Google Sheet. Check sharing + worksheet name.")
        st.stop()

try:
    df = load_and_normalize(df_raw)
except Exception as e:
    st.error(str(e))
    st.stop()

# Display conversion
if display_currency == "SEK":
    df["amount_display"] = df["amount_sek"].where(df["amount_sek"].notna(), df["amount_usd"] * fx)
else:
    df["amount_display"] = df["amount_usd"].where(df["amount_usd"].notna(), df["amount_sek"] / fx)

# -----------------------------
# Latest snapshot
# -----------------------------
latest_date = df["date"].max()
snap = df[df["date"] == latest_date].copy()

# Debt counts negative in net worth math
snap["signed_amount"] = snap["amount_display"]
snap.loc[snap["category"] == "Debt", "signed_amount"] = -snap.loc[snap["category"] == "Debt", "amount_display"]

def is_in_networth(row):
    return not (row["category"] == "Pension" and not include_pension_in_networth)

snap["in_networth"] = snap.apply(is_in_networth, axis=1)
current_networth = snap.loc[snap["in_networth"], "signed_amount"].sum()

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
start_debt = start_value("Debt")          # stored positive; subtract in net worth
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

def compute_snapshot_networth(df_snap: pd.DataFrame) -> float:
    df_snap = df_snap.copy()

    if display_currency == "SEK":
        df_snap["amount_display"] = df_snap["amount_sek"].where(df_snap["amount_sek"].notna(), df_snap["amount_usd"] * fx)
    else:
        df_snap["amount_display"] = df_snap["amount_usd"].where(df_snap["amount_usd"].notna(), df_snap["amount_sek"] / fx)

    df_snap["signed_amount"] = df_snap["amount_display"]
    df_snap.loc[df_snap["category"] == "Debt", "signed_amount"] = -df_snap.loc[df_snap["category"] == "Debt", "amount_display"]

    # pension default off
    df_snap["in_networth"] = df_snap.apply(is_in_networth, axis=1)
    return float(df_snap.loc[df_snap["in_networth"], "signed_amount"].sum())

# Utveckling (actual snapshots)
actual_rows = []
for d in sorted(df["date"].dropna().unique()):
    nw = compute_snapshot_networth(df[df["date"] == d])
    actual_rows.append({"date": pd.to_datetime(d), "value": nw, "series": "Utveckling"})
actual_df = pd.DataFrame(actual_rows)

# Prognos (forecast from latest snapshot onward)
forecast_df = proj.copy()
forecast_df["date"] = pd.to_datetime(forecast_df["date"])
forecast_df = forecast_df[["date", "Net Worth"]].rename(columns={"Net Worth": "value"})
forecast_df["series"] = "Prognos"

# Sparplan = straight line from earliest snapshot, NO return, just monthly savings
first_date = actual_df["date"].min().date()
first_nw = compute_snapshot_networth(df[df["date"] == first_date])

plan_dates = [month_add(first_date, int(m)) for m in months]
plan_df = pd.DataFrame({
    "date": pd.to_datetime(plan_dates),
    "value": first_nw + (monthly_savings * months),
    "series": "Sparplan",
})

plot_df = pd.concat([actual_df, forecast_df, plan_df], ignore_index=True)

chart = (
    alt.Chart(plot_df)
    .mark_line()
    .encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title=f"Värde ({display_currency})"),
        color=alt.Color("series:N", sort=["Utveckling", "Prognos", "Sparplan"], legend=alt.Legend(title=None)),
        strokeDash=alt.StrokeDash(
            "series:N",
            sort=["Utveckling", "Prognos", "Sparplan"],
            scale=alt.Scale(domain=["Utveckling", "Prognos", "Sparplan"], range=[[], [6, 4], [2, 2]]),
            legend=None,
        ),
        tooltip=["series:N", "date:T", alt.Tooltip("value:Q", format=",.0f")],
    )
    .properties(height=360)
)

st.altair_chart(chart, use_container_width=True)

# Goal ETA based on FORECAST net worth
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