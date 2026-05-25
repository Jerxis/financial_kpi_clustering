# src/data/financials_loader.py

from pathlib import Path
import time
import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm


def flatten_latest_financials(
    symbol: str,
    company_name: str,
    exchange: str,
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
) -> dict:
    """
    Flatten the latest available annual financial statements for one company.
    """

    data = {
        "Symbol": symbol,
        "Company Name": company_name,
        "Exchange": exchange,
    }

    if not income_stmt.empty:
        for metric, value in income_stmt.iloc[:, 0].items():
            data[f"Income_{metric}"] = value

    if not balance_sheet.empty:
        for metric, value in balance_sheet.iloc[:, 0].items():
            data[f"Balance_{metric}"] = value

    if not cash_flow.empty:
        for metric, value in cash_flow.iloc[:, 0].items():
            data[f"CashFlow_{metric}"] = value

    return data


def fetch_and_flatten_financials(
    ticker_universe: pd.DataFrame,
    output_dir: str | Path,
    sleep_seconds: float = 0.75,
    save_every: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch latest annual financial statements from yfinance and flatten them
    into one row per company.

    Saves intermediate checkpoint files to reduce risk of data loss.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flattened_records = []
    log_records = []

    total_companies = len(ticker_universe)

    for i, row in tqdm(
        ticker_universe.iterrows(),
        total=total_companies,
        desc="Fetching financial statements"
    ):
        symbol = row["Symbol"]
        company_name = row["Company Name"]
        exchange = row["Exchange"]

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1:,} of {total_companies:,} companies...")

        try:
            ticker = yf.Ticker(symbol)

            income_stmt = ticker.income_stmt
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cash_flow

            flattened_row = flatten_latest_financials(
                symbol=symbol,
                company_name=company_name,
                exchange=exchange,
                income_stmt=income_stmt,
                balance_sheet=balance_sheet,
                cash_flow=cash_flow,
            )

            flattened_records.append(flattened_row)

            log_records.append({
                "Symbol": symbol,
                "Company Name": company_name,
                "Exchange": exchange,
                "Status": "Success",
                "Error": None,
                "Income Statement Rows": income_stmt.shape[0],
                "Balance Sheet Rows": balance_sheet.shape[0],
                "Cash Flow Rows": cash_flow.shape[0],
            })

        except Exception as error:
            log_records.append({
                "Symbol": symbol,
                "Company Name": company_name,
                "Exchange": exchange,
                "Status": "Failed",
                "Error": str(error),
                "Income Statement Rows": None,
                "Balance Sheet Rows": None,
                "Cash Flow Rows": None,
            })

        if len(flattened_records) > 0 and len(flattened_records) % save_every == 0:
            checkpoint_df = pd.DataFrame(flattened_records)
            checkpoint_df.to_excel(
                output_dir / "financials_flattened_checkpoint.xlsx",
                index=False
            )

            log_df = pd.DataFrame(log_records)
            log_df.to_excel(
                output_dir / "financials_fetch_log_checkpoint.xlsx",
                index=False
            )

        time.sleep(sleep_seconds)

    financials_flat_df = pd.DataFrame(flattened_records)
    fetch_log_df = pd.DataFrame(log_records)

    financials_flat_df.to_excel(
        output_dir / "financials_flattened_raw.xlsx",
        index=False
    )

    fetch_log_df.to_excel(
        output_dir / "financials_fetch_log.xlsx",
        index=False
    )

    return financials_flat_df, fetch_log_df


def filter_companies_with_positive_revenue(
    financials_df: pd.DataFrame,
    revenue_column: str = "Income_Total Revenue",
) -> pd.DataFrame:
    """
    Keep only companies with positive total revenue.
    """

    filtered_df = financials_df[
        financials_df[revenue_column].notna()
        & (financials_df[revenue_column] > 0)
    ].copy()

    return filtered_df