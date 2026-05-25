# src/data/yfinance_loader.py

from pathlib import Path
import time
import pandas as pd
import yfinance as yf


def fetch_company_raw_data(ticker: str) -> dict:
    """
    Pull raw Yahoo Finance data for a single ticker.
    Returns a dictionary of raw DataFrames / dictionaries.
    """

    stock = yf.Ticker(ticker)

    return {
        "ticker": ticker,
        "info": stock.info,
        "income_statement": stock.financials,
        "balance_sheet": stock.balance_sheet,
        "cash_flow": stock.cashflow,
        "quarterly_income_statement": stock.quarterly_financials,
        "quarterly_balance_sheet": stock.quarterly_balance_sheet,
        "quarterly_cash_flow": stock.quarterly_cashflow,
    }


def save_company_raw_data(raw_data: dict, output_dir: str | Path) -> None:
    """
    Save raw company data into separate files.
    """

    ticker = raw_data["ticker"]
    company_dir = Path(output_dir) / ticker
    company_dir.mkdir(parents=True, exist_ok=True)

    # Save info dictionary
    pd.Series(raw_data["info"]).to_csv(company_dir / "info.csv")

    # Save financial statements
    for key, value in raw_data.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(company_dir / f"{key}.csv")


def fetch_multiple_companies(
    tickers: list[str],
    output_dir: str | Path = "data/raw/yfinance",
    sleep_seconds: float = 1.0
) -> pd.DataFrame:
    """
    Pull raw data for multiple companies and log success/failure.
    """

    log_records = []

    for ticker in tickers:
        try:
            print(f"Fetching {ticker}...")

            raw_data = fetch_company_raw_data(ticker)
            save_company_raw_data(raw_data, output_dir)

            log_records.append({
                "ticker": ticker,
                "status": "success",
                "error": None
            })

        except Exception as error:
            log_records.append({
                "ticker": ticker,
                "status": "failed",
                "error": str(error)
            })

        time.sleep(sleep_seconds)

    log_df = pd.DataFrame(log_records)
    log_df.to_csv(Path(output_dir) / "fetch_log.csv", index=False)

    return log_df