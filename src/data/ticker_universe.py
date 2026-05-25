# src/data/ticker_universe.py

from io import StringIO
from pathlib import Path
import pandas as pd
import requests


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"


EXCHANGE_MAPPING = {
    "Q": "NASDAQ",
    "G": "NASDAQ",
    "S": "NASDAQ",
    "N": "NYSE",
    "P": "NYSE",
    "A": "AMEX",
}


EXCLUDED_SECURITY_NAME_KEYWORDS = [
    "ETF",
    "Fund",
    "Trust",
    "Bond",
    "Note",
    "ETN",
    "Preferred",
    "Preference",
    "Warrant",
    "Right",
    "Unit",
    "Units",
    "Index",
    "Portfolio",
    "Income",
    "Treasury",
    "Municipal",
]


def download_symbol_directory(url: str) -> pd.DataFrame:
    """
    Download a NasdaqTrader symbol directory file.
    """

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text), sep="|")

    # Remove metadata row
    first_column = df.columns[0]
    df = df[~df[first_column].astype(str).str.contains("File Creation Time", na=False)]

    return df


def clean_nasdaq_listings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Nasdaq-listed securities.
    """

    cleaned = df.copy()

    cleaned = cleaned.dropna(subset=["Symbol"])

    cleaned = cleaned[
        (cleaned["ETF"] == "N")
        & (cleaned["Test Issue"] == "N")
        & (cleaned["NextShares"] == "N")
    ]

    cleaned["Exchange"] = cleaned["Market Category"].map(EXCHANGE_MAPPING).fillna("NASDAQ")

    cleaned = cleaned.rename(columns={"Security Name": "Company Name"})

    return cleaned[["Symbol", "Company Name", "Exchange"]]


def clean_other_listings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NYSE and AMEX listings from NasdaqTrader's other-listed securities file.
    """

    cleaned = df.copy()

    cleaned = cleaned.dropna(subset=["ACT Symbol"])

    cleaned = cleaned[
        (cleaned["ETF"] == "N")
        & (cleaned["Test Issue"] == "N")
    ]

    cleaned["Exchange"] = cleaned["Exchange"].map(EXCHANGE_MAPPING)

    cleaned = cleaned[cleaned["Exchange"].isin(["NYSE", "AMEX"])]

    cleaned = cleaned.rename(
        columns={
            "ACT Symbol": "Symbol",
            "Security Name": "Company Name",
        }
    )

    return cleaned[["Symbol", "Company Name", "Exchange"]]


def remove_likely_non_operating_securities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove securities that are unlikely to represent operating companies.
    This is intentionally conservative and can be refined later.
    """

    cleaned = df.copy()

    pattern = "|".join(EXCLUDED_SECURITY_NAME_KEYWORDS)

    cleaned = cleaned[
        ~cleaned["Company Name"].str.contains(pattern, case=False, na=False)
    ]

    return cleaned


def build_us_equity_universe(
    remove_non_operating_securities: bool = True,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build a combined NASDAQ, NYSE, and AMEX ticker universe.
    """

    nasdaq_raw = download_symbol_directory(NASDAQ_LISTED_URL)
    other_raw = download_symbol_directory(OTHER_LISTED_URL)

    nasdaq_clean = clean_nasdaq_listings(nasdaq_raw)
    other_clean = clean_other_listings(other_raw)

    universe = pd.concat([nasdaq_clean, other_clean], ignore_index=True)

    universe = universe.drop_duplicates(subset=["Symbol"])
    universe = universe.sort_values(["Exchange", "Symbol"]).reset_index(drop=True)

    if remove_non_operating_securities:
        universe = remove_likely_non_operating_securities(universe)
        universe = universe.reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        universe.to_excel(output_path, index=False)

    return universe