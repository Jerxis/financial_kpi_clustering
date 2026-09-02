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


def make_safe_column_name(column_name: str) -> str:
    """
    Convert a column name into a safe variable name for formula evaluation.
    """

    return (
        column_name
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("&", "and")
        .replace("(", "")
        .replace(")", "")
    )


def make_safe_formula(formula: str, column_mapping: dict) -> str:
    """
    Replace original column names in a formula with safe column names.
    """

    safe_formula = formula

    for original_column, safe_column in column_mapping.items():
        safe_formula = safe_formula.replace(f"`{original_column}`", safe_column)

    return safe_formula


def calculate_single_kpi(
    data: pd.DataFrame,
    kpi_name: str,
    formula: str
) -> pd.Series:
    """
    Calculate a single KPI using safe column names.
    """

    column_mapping = {
        column: make_safe_column_name(column)
        for column in data.columns
    }

    safe_data = data.rename(columns=column_mapping)

    safe_formula = make_safe_formula(
        formula=formula,
        column_mapping=column_mapping
    )

    local_dict = {
        col: safe_data[col]
        for col in safe_data.columns
    }

    local_dict["np"] = np
    local_dict["pd"] = pd

    try:
        result = pd.eval(
            safe_formula,
            local_dict=local_dict,
            engine="python"
        )

        result = result.replace([np.inf, -np.inf], np.nan)

        return result

    except Exception as error:
        raise ValueError(
            f"Failed to calculate KPI '{kpi_name}'. "
            f"Formula used: {safe_formula}. "
            f"Error: {error}"
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


def diagnose_kpi_economic_validity(
    data: pd.DataFrame,
    minimum_positive_net_income_margin: float = 0.01,
) -> pd.DataFrame:
    """Return one audit row per economically invalid KPI observation.

    These rules do not claim that extreme financial performance is an error.
    They only identify observations for which a ratio is mathematically unsafe
    or loses its usual economic interpretation because a required denominator
    or input is missing, non-positive, or immaterial relative to revenue.

    Cash Conversion Ratio receives the only materiality rule: net income must
    be positive and at least ``minimum_positive_net_income_margin`` of revenue.
    The threshold is therefore explicit and can be varied in sensitivity tests.
    """

    identity_columns = ["Symbol", "Company Name", "Sector"]
    required_identity_columns = [
        column for column in identity_columns if column not in data.columns
    ]
    if required_identity_columns:
        raise ValueError(
            "Missing identity columns for KPI validity audit: "
            f"{required_identity_columns}"
        )

    rule_specs = [
        {
            "kpis": [
                "EBITDA Margin",
                "Free Cash Flow Margin",
                "Operating Cash Flow Margin",
            ],
            "field": "Total Revenue",
            "reason": "non_positive_denominator",
            "description": "Total Revenue must be greater than zero.",
            "invalid": lambda values, frame: values <= 0,
        },
        {
            "kpis": ["Return on Assets", "Asset Turnover", "Debt Ratio"],
            "field": "Total Assets",
            "reason": "non_positive_denominator",
            "description": "Total Assets must be greater than zero.",
            "invalid": lambda values, frame: values <= 0,
        },
        {
            "kpis": ["Revenue per Employee"],
            "field": "Employees",
            "reason": "non_positive_denominator",
            "description": "Employee count must be greater than zero.",
            "invalid": lambda values, frame: values <= 0,
        },
        {
            "kpis": ["Current Ratio"],
            "field": "Current Liabilities",
            "reason": "non_positive_denominator",
            "description": "Current Liabilities must be greater than zero.",
            "invalid": lambda values, frame: values <= 0,
        },
        {
            "kpis": ["Debt Ratio"],
            "field": "Total Debt",
            "reason": "negative_numerator",
            "description": "Total Debt must not be negative.",
            "invalid": lambda values, frame: values < 0,
        },
        {
            "kpis": ["Cash Conversion Ratio"],
            "field": "Net Income",
            "reason": "non_positive_denominator",
            "description": (
                "Cash Conversion Ratio is not given its usual interpretation "
                "when Net Income is zero or negative."
            ),
            "invalid": lambda values, frame: values <= 0,
        },
        {
            "kpis": ["Cash Conversion Ratio"],
            "field": "Net Income",
            "reason": "immaterial_positive_denominator",
            "description": (
                "Positive Net Income must be material relative to revenue."
            ),
            "invalid": lambda values, frame: (
                (values > 0)
                & frame["Total Revenue"].gt(0)
                & (
                    values / frame["Total Revenue"]
                    < minimum_positive_net_income_margin
                )
            ),
        },
    ]

    required_fields = sorted(
        {spec["field"] for spec in rule_specs} | {"Total Revenue"}
    )
    missing_fields = [
        field for field in required_fields if field not in data.columns
    ]
    if missing_fields:
        raise ValueError(
            "Missing financial fields for KPI validity audit: "
            f"{missing_fields}"
        )

    audit_records = []
    audited_field_kpis = set()

    for spec in rule_specs:
        field = spec["field"]
        field_values = data[field]

        for kpi in spec["kpis"]:
            field_kpi_key = (field, kpi)
            if field_kpi_key not in audited_field_kpis:
                missing_mask = field_values.isna()
                for row_index in data.index[missing_mask]:
                    audit_records.append({
                        **data.loc[row_index, identity_columns].to_dict(),
                        "KPI": kpi,
                        "Reason": "source_field_missing",
                        "Rule Description": f"{field} is required.",
                        "Rule Field": field,
                        "Observed Value": np.nan,
                        "Reference Value": np.nan,
                        "Materiality Threshold": np.nan,
                    })
                audited_field_kpis.add(field_kpi_key)

            invalid_mask = (
                field_values.notna()
                & spec["invalid"](field_values, data)
            )
            for row_index in data.index[invalid_mask]:
                audit_records.append({
                    **data.loc[row_index, identity_columns].to_dict(),
                    "KPI": kpi,
                    "Reason": spec["reason"],
                    "Rule Description": spec["description"],
                    "Rule Field": field,
                    "Observed Value": field_values.loc[row_index],
                    "Reference Value": (
                        data.loc[row_index, "Total Revenue"]
                        if kpi == "Cash Conversion Ratio"
                        else np.nan
                    ),
                    "Materiality Threshold": (
                        minimum_positive_net_income_margin
                        if spec["reason"]
                        == "immaterial_positive_denominator"
                        else np.nan
                    ),
                })

    audit_columns = [
        *identity_columns,
        "KPI",
        "Reason",
        "Rule Description",
        "Rule Field",
        "Observed Value",
        "Reference Value",
        "Materiality Threshold",
    ]
    return pd.DataFrame(audit_records, columns=audit_columns)


def apply_kpi_economic_validity_mask(
    kpi_data: pd.DataFrame,
    validity_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Set audited invalid KPI observations to missing, preserving provenance."""

    if kpi_data["Symbol"].duplicated().any():
        raise ValueError("KPI validity masking requires one row per Symbol.")

    masked_data = kpi_data.copy()
    symbol_to_index = pd.Series(masked_data.index, index=masked_data["Symbol"])

    invalid_pairs = validity_audit[["Symbol", "KPI"]].drop_duplicates()
    unknown_kpis = sorted(set(invalid_pairs["KPI"]) - set(masked_data.columns))
    if unknown_kpis:
        raise ValueError(f"Validity audit contains unknown KPIs: {unknown_kpis}")

    for kpi, invalid_symbols in invalid_pairs.groupby("KPI")["Symbol"]:
        matching_symbols = invalid_symbols[invalid_symbols.isin(symbol_to_index.index)]
        masked_data.loc[symbol_to_index.loc[matching_symbols], kpi] = np.nan

    return masked_data
