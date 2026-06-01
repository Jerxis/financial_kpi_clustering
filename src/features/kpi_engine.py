from pathlib import Path
import pandas as pd
import numpy as np


def load_kpi_definitions(file_path: str | Path) -> pd.DataFrame:
    """
    Load KPI definitions from a CSV file.
    """

    kpi_definitions = pd.read_csv(file_path)

    required_columns = [
        "ID",
        "KPI",
        "Category",
        "UoM",
        "What is better?",
        "Formula",
        "Required Fields",
        "Include in Clustering",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in kpi_definitions.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return kpi_definitions


def parse_required_fields(required_fields: str) -> list[str]:
    """
    Convert semicolon-separated required fields into a list.
    """

    if pd.isna(required_fields):
        return []

    return [
        field.strip()
        for field in required_fields.split(";")
        if field.strip()
    ]


def validate_kpi_definitions(
    data: pd.DataFrame,
    kpi_definitions: pd.DataFrame
) -> pd.DataFrame:
    """
    Check whether required fields for each KPI exist in the dataset.
    """

    validation_records = []

    for _, row in kpi_definitions.iterrows():
        required_fields = parse_required_fields(row["Required Fields"])

        missing_fields = [
            field for field in required_fields
            if field not in data.columns
        ]

        validation_records.append({
            "KPI": row["KPI"],
            "Required Fields": required_fields,
            "Missing Fields": missing_fields,
            "Can Calculate": len(missing_fields) == 0,
        })

    return pd.DataFrame(validation_records)


def calculate_single_kpi(
    data: pd.DataFrame,
    kpi_name: str,
    formula: str
) -> pd.Series:
    """
    Calculate a single KPI using a restricted evaluation environment.
    """

    local_dict = {
        col: data[col]
        for col in data.columns
    }

    local_dict["np"] = np
    local_dict["pd"] = pd

    try:
        result = pd.eval(
            formula,
            local_dict=local_dict,
            engine="python"
        )

        result = result.replace([np.inf, -np.inf], np.nan)

        return result

    except Exception as error:
        raise ValueError(
            f"Failed to calculate KPI '{kpi_name}'. Error: {error}"
        )


def calculate_kpis(
    data: pd.DataFrame,
    kpi_definitions: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate all KPIs defined in the KPI definitions file.
    """

    kpi_results = data[[
        "Symbol",
        "Company Name",
        "Exchange",
        "Sector",
        "Industry",
    ]].copy()

    for _, row in kpi_definitions.iterrows():
        kpi_name = row["KPI"]
        formula = row["Formula"]

        kpi_results[kpi_name] = calculate_single_kpi(
            data=data,
            kpi_name=kpi_name,
            formula=formula
        )

    return kpi_results