"""Diagnostics for sector, industry, and company-size confounding."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import RobustScaler, StandardScaler


def calculate_bias_corrected_cramers_v(
    labels: np.ndarray,
    categories: pd.Series,
) -> dict:
    """Measure categorical association with sparse-table diagnostics."""

    aligned = pd.DataFrame({
        "Cluster": np.asarray(labels),
        "Category": pd.Series(categories).reset_index(drop=True),
    }).dropna()
    contingency = pd.crosstab(aligned["Cluster"], aligned["Category"])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {
            "Bias-Corrected Cramer's V": np.nan,
            "Adjusted Mutual Information": np.nan,
            "Chi-Square P-Value": np.nan,
            "Contingency Rows": contingency.shape[0],
            "Contingency Columns": contingency.shape[1],
            "Expected Cells Below 5 Share": np.nan,
            "N Evaluated": len(aligned),
        }

    chi_square, p_value, _, expected = chi2_contingency(
        contingency,
        correction=False,
    )
    n_observations = contingency.to_numpy().sum()
    phi_squared = chi_square / n_observations
    n_rows, n_columns = contingency.shape

    corrected_phi_squared = max(
        0,
        phi_squared
        - ((n_columns - 1) * (n_rows - 1))
        / (n_observations - 1),
    )
    corrected_rows = (
        n_rows - ((n_rows - 1) ** 2) / (n_observations - 1)
    )
    corrected_columns = (
        n_columns
        - ((n_columns - 1) ** 2) / (n_observations - 1)
    )
    denominator = min(
        corrected_columns - 1,
        corrected_rows - 1,
    )
    cramers_v = (
        np.sqrt(corrected_phi_squared / denominator)
        if denominator > 0
        else np.nan
    )

    return {
        "Bias-Corrected Cramer's V": float(cramers_v),
        "Adjusted Mutual Information": float(
            adjusted_mutual_info_score(
                aligned["Cluster"],
                aligned["Category"],
            )
        ),
        "Chi-Square P-Value": float(p_value),
        "Contingency Rows": n_rows,
        "Contingency Columns": n_columns,
        "Expected Cells Below 5 Share": float((expected < 5).mean()),
        "N Evaluated": len(aligned),
    }


def calculate_continuous_cluster_association(
    labels: np.ndarray,
    values: pd.Series,
) -> dict:
    """Measure continuous-variable differences across cluster labels."""

    aligned = pd.DataFrame({
        "Cluster": np.asarray(labels),
        "Value": pd.to_numeric(
            pd.Series(values).reset_index(drop=True),
            errors="coerce",
        ),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    groups = [
        group["Value"].to_numpy()
        for _, group in aligned.groupby("Cluster")
        if len(group) > 0
    ]

    if len(groups) < 2 or len(aligned) <= len(groups):
        return {
            "Kruskal-Wallis H": np.nan,
            "Kruskal-Wallis P-Value": np.nan,
            "Epsilon-Squared": np.nan,
            "Cluster Median Range": np.nan,
            "N Evaluated": len(aligned),
            "Evaluation Coverage": len(aligned) / len(labels),
        }

    h_statistic, p_value = kruskal(*groups)
    epsilon_squared = max(
        0,
        (
            h_statistic - len(groups) + 1
        ) / (len(aligned) - len(groups)),
    )
    cluster_medians = aligned.groupby("Cluster")["Value"].median()

    return {
        "Kruskal-Wallis H": float(h_statistic),
        "Kruskal-Wallis P-Value": float(p_value),
        "Epsilon-Squared": float(epsilon_squared),
        "Cluster Median Range": float(
            cluster_medians.max() - cluster_medians.min()
        ),
        "N Evaluated": len(aligned),
        "Evaluation Coverage": len(aligned) / len(labels),
    }


def benjamini_hochberg_adjust(
    p_values: pd.Series,
) -> pd.Series:
    """Adjust a family of p-values while preserving missing entries."""

    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)

    if valid.empty:
        return adjusted

    ordered = valid.sort_values()
    ranks = np.arange(1, len(ordered) + 1)
    ordered_adjusted = (
        ordered.to_numpy() * len(ordered) / ranks
    )
    ordered_adjusted = np.minimum.accumulate(
        ordered_adjusted[::-1]
    )[::-1]
    ordered_adjusted = np.clip(ordered_adjusted, 0, 1)
    adjusted.loc[ordered.index] = ordered_adjusted

    return adjusted


def residualize_features_against_confounders(
    X: np.ndarray,
    metadata: pd.DataFrame,
    revenue_column: str = "Total Revenue",
    market_cap_column: str = "MarketCap",
    sector_column: str = "Sector",
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Remove linear size and, where applicable, sector associations.

    Residualisation is a sensitivity diagnostic, not a replacement for the
    canonical economic KPI space. Market-cap missingness is represented
    explicitly and missing log market capitalisation is median-imputed.
    """

    validated_X = np.asarray(X, dtype=float)
    if validated_X.ndim != 2 or len(validated_X) != len(metadata):
        raise ValueError("X and metadata must have aligned observations.")
    if not np.isfinite(validated_X).all():
        raise ValueError("X must contain only finite values.")

    required_columns = [
        revenue_column,
        market_cap_column,
        sector_column,
    ]
    missing_columns = [
        column for column in required_columns
        if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing confounder columns: {missing_columns}"
        )

    revenue = pd.to_numeric(
        metadata[revenue_column],
        errors="coerce",
    )
    market_cap = pd.to_numeric(
        metadata[market_cap_column],
        errors="coerce",
    )
    if revenue.le(0).any() or revenue.isna().any():
        raise ValueError(
            "Revenue must be positive and complete for residualisation."
        )

    log_market_cap = np.log10(market_cap.where(market_cap > 0))
    market_cap_missing = log_market_cap.isna().astype(int)
    market_cap_median = log_market_cap.median()
    if pd.isna(market_cap_median):
        market_cap_median = 0.0
    log_market_cap = log_market_cap.fillna(market_cap_median)

    confounders = pd.DataFrame({
        "Log10 Revenue": np.log10(revenue),
        "Log10 MarketCap": log_market_cap,
        "MarketCap Missing": market_cap_missing,
    }).reset_index(drop=True)

    if metadata[sector_column].nunique(dropna=True) > 1:
        sector_dummies = pd.get_dummies(
            metadata[sector_column].fillna("Missing"),
            prefix="Sector",
            drop_first=True,
            dtype=float,
        ).reset_index(drop=True)
        confounders = pd.concat(
            [confounders, sector_dummies],
            axis=1,
        )

    constant_confounders = [
        column
        for column in confounders
        if confounders[column].nunique(dropna=False) <= 1
    ]
    confounders = confounders.drop(columns=constant_confounders)

    confounder_array = StandardScaler().fit_transform(confounders)
    residuals = np.empty_like(validated_X, dtype=float)
    explained_variance_records = []

    for feature_index in range(validated_X.shape[1]):
        model = TransformedTargetRegressor(
            regressor=Ridge(alpha=ridge_alpha),
            transformer=StandardScaler(),
        )
        model.fit(confounder_array, validated_X[:, feature_index])
        predictions = model.predict(confounder_array)
        residuals[:, feature_index] = (
            validated_X[:, feature_index] - predictions
        )
        explained_variance_records.append({
            "Feature Index": feature_index,
            "Confounder Model R-Squared": model.score(
                confounder_array,
                validated_X[:, feature_index],
            ),
        })

    scaled_residuals = RobustScaler().fit_transform(residuals)

    return (
        scaled_residuals,
        pd.DataFrame(explained_variance_records),
        confounders,
    )
