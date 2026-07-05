import pandas as pd


def filter_peer_universe(
    data: pd.DataFrame,
    sectors: list[str] | None = None,
    revenue_column: str = "Total Revenue",
    revenue_range: tuple[float, float] | None = None,
    exclude_revenue_outliers: bool = False,
) -> pd.DataFrame:
    """
    Filter the company universe before clustering.
    """

    filtered_df = data.copy()

    if sectors:
        filtered_df = filtered_df[
            filtered_df["Sector"].isin(sectors)
        ].copy()

    if exclude_revenue_outliers:
        q1 = filtered_df[revenue_column].quantile(0.25)
        q3 = filtered_df[revenue_column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_df = filtered_df[
            (filtered_df[revenue_column] >= lower_bound)
            & (filtered_df[revenue_column] <= upper_bound)
        ].copy()

    if revenue_range is not None:
        min_revenue, max_revenue = revenue_range

        filtered_df = filtered_df[
            (filtered_df[revenue_column] >= min_revenue)
            & (filtered_df[revenue_column] <= max_revenue)
        ].copy()

    return filtered_df.reset_index(drop=True)