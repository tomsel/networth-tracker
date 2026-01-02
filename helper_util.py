import streamlit as st
import os
import re
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials


# -----------------------------
# General utils
# -----------------------------
def parse_date_col(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def safe_float_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "None": np.nan, "nan": np.nan})
        .astype(float)
    )


def month_add(d, m: int):
    return (pd.Period(d, freq="M") + int(m)).to_timestamp().date()


def fmt_money(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


# -----------------------------
# Categories / schema
# -----------------------------
VALID_CATEGORIES = {"Cash", "Investments", "Debt", "Real Estate", "Pension"}

BASE_COLS_MAIN = ["date", "owner", "category", "amount_sek", "amount_usd"]
BASE_COLS_STATIC = ["owner", "category", "amount_sek", "amount_usd"]


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


# -----------------------------
# Normalization
# -----------------------------
def load_and_normalize_main(df_raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in BASE_COLS_MAIN if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Main missing columns: {missing}")

    df = df_raw.copy()
    df["date"] = parse_date_col(df["date"])
    df = df.dropna(subset=["date"])

    df["owner"] = df["owner"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).apply(canonical_category)

    df["amount_sek"] = safe_float_series(df["amount_sek"])
    df["amount_usd"] = safe_float_series(df["amount_usd"])

    return df


def load_and_normalize_static(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=BASE_COLS_STATIC)

    missing = [c for c in BASE_COLS_STATIC if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Static missing columns: {missing}")

    df = df_raw.copy()
    df["owner"] = df["owner"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).apply(canonical_category)

    df["amount_sek"] = safe_float_series(df["amount_sek"])
    df["amount_usd"] = safe_float_series(df["amount_usd"])

    return df


def add_amount_display(df: pd.DataFrame, display_currency: str, fx: float) -> pd.DataFrame:
    df = df.copy()
    if display_currency == "SEK":
        df["amount_display"] = df["amount_sek"].where(
            df["amount_sek"].notna(), df["amount_usd"] * fx
        )
    else:
        df["amount_display"] = df["amount_usd"].where(
            df["amount_usd"].notna(), df["amount_sek"] / fx
        )
    return df


def combine_snapshot(main_snap: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    if static_df is None or static_df.empty:
        return main_snap.copy()

    s = static_df.copy()
    m = main_snap.copy()

    s["_k"] = s["owner"] + "||" + s["category"]
    m["_k"] = m["owner"] + "||" + m["category"]

    s_keep = s[~s["_k"].isin(set(m["_k"]))].drop(columns="_k")
    m = m.drop(columns="_k")

    return pd.concat([m, s_keep], ignore_index=True)


# -----------------------------
# I/O
# -----------------------------

def _extract_sheet_key(url_or_key: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url_or_key)
    return m.group(1) if m else url_or_key


def read_google_sheet_to_df(sheet_url: str, worksheet: str, sa_json_path: str | None):
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    if sa_json_path and os.path.exists(sa_json_path):
        creds = Credentials.from_service_account_file(sa_json_path, scopes=scopes)
    else:
        # For Streamlit Cloud: put keys in secrets.toml under [gcp_service_account]
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