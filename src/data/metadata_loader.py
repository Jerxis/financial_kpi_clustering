# src/data/metadata_loader.py

from pathlib import Path
import time

import pandas as pd
import yfinance as yf

from tqdm.auto import tqdm


METADATA_FIELDS = {
    "sector": "Sector",
    "industry": "Industry",
    "country": "Country",
    "fullTimeEmployees": "Employees",
    "marketCap": "MarketCap",
    "enterpriseValue": "EnterpriseValue",
    "currency": "Currency",
    "quoteType": "QuoteType",
    "website": "Website",
}


def fetch_company_metadata(symbol: str) -> dict:
    """
    Fetch metadata for a single company.
    """

    metadata = {"Symbol": symbol}

    try:
        ticker = yf.Ticker(symbol)

        info = ticker.info

        for yahoo_field, output_field in METADATA_FIELDS.items():
            metadata[output_field] = info.get(yahoo_field)

        metadata["Status"] = "Success"
        metadata["Error"] = None

    except Exception as error:
        metadata["Status"] = "Failed"
        metadata["Error"] = str(error)

        for output_field in METADATA_FIELDS.values():
            metadata[output_field] = None

    return metadata


def fetch_metadata_batch(
    ticker_universe: pd.DataFrame,
    output_dir: str | Path,
    sleep_seconds: float = 0.75,
    save_every: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch company metadata for a ticker universe.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_records = []

    total_companies = len(ticker_universe)

    for i, row in tqdm(
        ticker_universe.iterrows(),
        total=total_companies,
        desc="Fetching company metadata"
    ):
        symbol = row["Symbol"]

        metadata = fetch_company_metadata(symbol)

        metadata_records.append(metadata)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1:,} of {total_companies:,} companies...")

        if len(metadata_records) > 0 and len(metadata_records) % save_every == 0:
            checkpoint_df = pd.DataFrame(metadata_records)

            checkpoint_df.to_excel(
                output_dir / "metadata_checkpoint.xlsx",
                index=False
            )

        time.sleep(sleep_seconds)

    metadata_df = pd.DataFrame(metadata_records)

    metadata_df.to_excel(
        output_dir / "company_metadata.xlsx",
        index=False
    )

    failed_metadata_df = metadata_df[
        metadata_df["Status"] != "Success"
    ]

    failed_metadata_df.to_excel(
        output_dir / "metadata_failures.xlsx",
        index=False
    )

    return metadata_df, failed_metadata_df