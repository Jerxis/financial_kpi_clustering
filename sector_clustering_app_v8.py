import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go

# ========== SETTINGS ==========
EXCEL_PATH = r"C:\Users\semoy\OneDrive\Documentos\Keele University\1.Computer Science Project\Project datasets\4.final_dataset_KPIs_v2.xlsx"
KPI_METADATA_PATH = r"C:\Users\semoy\OneDrive\Documentos\Keele University\1.Computer Science Project\Project datasets\0.KPI_formulas.xlsx"
SHEET_NAME = 0
KPI_FEATURES = [
    "Gross Margin",
    "Operating Margin",
    "EBITDA Margin",
    "Return on Assets",
    "Asset Turnover",
    "Revenue per Employee"
]

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    df.dropna(subset=["Sector"], inplace=True)
    return df

@st.cache_data
def load_kpi_metadata():
    metadata = pd.read_excel(KPI_METADATA_PATH, sheet_name=0)
    return metadata

df = load_data()
kpi_metadata = load_kpi_metadata()

# ==== SIDEBAR FILTERS ====
st.sidebar.title("Filter Options")

# --- Sector selection ---
st.sidebar.markdown("**Choose sector(s):**")
sectors = df["Sector"].dropna().unique()
selected_sectors = st.sidebar.multiselect("", sorted(sectors))

# --- Build sector-filtered dataset ---
if selected_sectors:
    sector_filtered_df = df[df["Sector"].isin(selected_sectors)]
else:
    sector_filtered_df = df.copy()

# --- Exclude outliers toggle ---
exclude_outliers = st.sidebar.checkbox("Exclude revenue outliers (IQR method)", value=False)

st.sidebar.markdown("---")

# --- Log scale toggle ---
use_log_scale = st.sidebar.checkbox("Logarithmic revenue slider", value=False)

# --- IQR calculation for outlier detection ---
Q1 = sector_filtered_df["Total Revenue"].quantile(0.25)
Q3 = sector_filtered_df["Total Revenue"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

analysis_df = sector_filtered_df.copy()
if exclude_outliers:
    analysis_df = analysis_df[
        (analysis_df["Total Revenue"] >= lower_bound) &
        (analysis_df["Total Revenue"] <= upper_bound)
    ]

# --- Helper for human readable labels ---
def human_format(num):
    if num >= 1_000_000_000:
        return f"${num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"${num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"${num/1_000:.1f}K"
    else:
        return f"${num:,.0f}"

# --- Revenue slider ---
if use_log_scale:
    min_rev = max(analysis_df["Total Revenue"].min(), 1)
    max_rev = analysis_df["Total Revenue"].max()
    log_range = st.sidebar.slider(
        "Revenue size (log10 USD)",
        min_value=float(np.log10(min_rev)),
        max_value=float(np.log10(max_rev)),
        value=(float(np.log10(min_rev)), float(np.log10(max_rev))),
        step=(np.log10(max_rev) - np.log10(min_rev)) / 100
    )
    revenue_range = (10 ** log_range[0], 10 ** log_range[1])
else:
    min_rev = int(analysis_df["Total Revenue"].min())
    max_rev = int(analysis_df["Total Revenue"].max())
    revenue_range = st.sidebar.slider(
        "Revenue size (USD)",
        min_value=min_rev,
        max_value=max_rev,
        value=(min_rev, max_rev),
        step=max(1, int((max_rev - min_rev) / 100))
    )

# --- Show formatted revenue range ---
st.sidebar.markdown(f"**Selected range:** {human_format(revenue_range[0])} → {human_format(revenue_range[1])}")

# --- Matching companies preview ---
preview_df = analysis_df[
    (analysis_df["Total Revenue"] >= revenue_range[0]) &
    (analysis_df["Total Revenue"] <= revenue_range[1])
]
preview_count = len(preview_df)
st.sidebar.markdown(f"**Matching companies:** {preview_count:,}")
if preview_count < 5:
    st.sidebar.warning("Few companies selected (<5)")

st.sidebar.markdown("---")

# --- Clustering parameter ---
st.sidebar.markdown("**Number of Clusters**")
n_clusters = st.sidebar.slider("", min_value=2, max_value=10, value=4, step=1)
apply_button = st.sidebar.button(label="Apply")


# ========== MAIN PANEL ==========
st.set_page_config(layout="wide")
st.title("Dynamic Sector Clustering Tool")

if not apply_button or not selected_sectors:
    st.info("Please select one or more sectors and click Apply.")
    st.stop()

# === Final filtering ===
filtered_df = preview_df.dropna(subset=KPI_FEATURES + ["Company Name"]).copy()
if filtered_df.empty:
    st.warning("No companies match the selected filters. Adjust filters.")
    st.stop()

if exclude_outliers:
    excluded_df = sector_filtered_df[
        (sector_filtered_df["Total Revenue"] < lower_bound) |
        (sector_filtered_df["Total Revenue"] > upper_bound)
    ].sort_values(by="Total Revenue", ascending=False)

    excluded_count = len(excluded_df)

    if excluded_count > 0:
        with st.expander(f"Excluded Companies (Outliers) — {excluded_count} companies", expanded=False):
            st.dataframe(
                excluded_df[["Company Name", "Sector", "Total Revenue"]],
                use_container_width=True,
                hide_index=True
            )
            
# ========= SCALE + PCA ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(filtered_df[KPI_FEATURES])

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)
filtered_df["PCA1"] = X_pca[:, 0]
filtered_df["PCA2"] = X_pca[:, 1]

# ========= OUTLIER REMOVAL ==========
z_scores = (filtered_df["PCA1"] - filtered_df["PCA1"].mean()) / filtered_df["PCA1"].std()
mask = np.abs(z_scores) < 3
filtered_df = filtered_df[mask].copy()
X_cluster = X_pca[mask]  # Match length

# ========= CLUSTER EVALUATION ==========
st.subheader("Cluster Evaluation Metrics")

# === Metric Definitions ===

with st.expander("ℹ️ Definitions: "):
    st.markdown("""
    - **K-Means Clustering**: A machine learning algorithm that groups data into *k* clusters by minimizing the variance within each cluster. It iteratively assigns points to the nearest cluster center and updates the centers.
    - **Elbow Method**: Plots the Within-Cluster Sum of Squares (WCSS) against the number of clusters. The 'elbow' point suggests a good balance between model complexity and performance.
    - **Silhouette Score**: Ranges from -1 to 1. A higher score means that clusters are well-separated and cohesive.
    - **Davies-Bouldin Index**: A lower score indicates better clustering. It considers both the compactness of clusters and the distance between them.
    - **PCA (Principal Component Analysis)**: A technique to reduce dimensionality by transforming features into a smaller set of uncorrelated components, making data easier to visualize and analyze.
    """)



wcss = []
sil_scores = []
dbi_scores = []
K_range = range(2, 11)

for k in K_range:
    model = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X_cluster)
    labels = model.labels_
    wcss.append(model.inertia_)
    sil_scores.append(silhouette_score(X_cluster, labels))
    dbi_scores.append(davies_bouldin_score(X_cluster, labels))

# Find best k
best_sil_k = K_range[np.argmax(sil_scores)]
best_dbi_k = K_range[np.argmin(dbi_scores)]

col1, col2, col3 = st.columns(3)

with col1:
    fig_wcss = px.line(x=list(K_range), y=wcss, markers=True, labels={"x": "Clusters", "y": "WCSS"}, title="Elbow Method (WCSS)")
    fig_wcss.add_vline(x=n_clusters, line_dash="dash", line_color="red")
    st.plotly_chart(fig_wcss, use_container_width=True)

with col2:
    fig_sil = px.line(x=list(K_range), y=sil_scores, markers=True, labels={"x": "Clusters", "y": "Silhouette Score"}, title="Silhouette Score")
    fig_sil.add_vline(x=n_clusters, line_dash="dash", line_color="red")
    st.plotly_chart(fig_sil, use_container_width=True)

with col3:
    fig_dbi = px.line(x=list(K_range), y=dbi_scores, markers=True, labels={"x": "Clusters", "y": "DBI (lower better)"}, title="Davies-Bouldin Index")
    fig_dbi.add_vline(x=n_clusters, line_dash="dash", line_color="red")
    st.plotly_chart(fig_dbi, use_container_width=True)

st.markdown(f"✅ **Suggested number of clusters based on Silhouette Score:** {best_sil_k}")

# ========= FINAL CLUSTERING ==========
kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
filtered_df["Cluster"] = kmeans.fit_predict(X_cluster)

# ========= TOP ROW: PLOT + METADATA ==========
st.markdown("## Overview")
top_left, top_right = st.columns([3, 2], gap="large")

with top_left:
    fig = px.scatter(
        filtered_df,
        x="PCA1",
        y="PCA2",
        color=filtered_df["Cluster"].astype(str),
        hover_data={
            "Company Name": True,
            "Sector": True,
            "Total Revenue": True,
            "Revenue per Employee": True,
            "Cluster": True
        },
        title="PCA + KMeans Clustering"
    )
    fig.update_layout(template="plotly_white", legend_title_text="Cluster", height=600)
    st.plotly_chart(fig, use_container_width=True)

with top_right:
    counts = filtered_df['Cluster'].value_counts().sort_index()
    count_df = pd.DataFrame({
        "Cluster": counts.index,
        "Company Count": counts.values
    })
    st.subheader("Company Count per Cluster")
    st.dataframe(count_df, use_container_width=True, hide_index=True)

    st.subheader("Sector Summary")
    st.markdown(f"**Total Companies:** {len(filtered_df)}")
    st.markdown(f"**Average Gross Margin:** {filtered_df['Gross Margin'].mean():.2%}")
    st.markdown(f"**Median Revenue per Employee:** ${filtered_df['Revenue per Employee'].median():,.0f}")

# ========= SCREE PLOT ==========
st.subheader("Explained Variance by Principal Component")
explained_var = pca.explained_variance_ratio_
fig_var = px.bar(
    x=[f'PC{i+1}' for i in range(len(explained_var))],
    y=explained_var,
    labels={'x': 'Principal Component', 'y': 'Explained Variance'},
    title='PCA Scree Plot'
)
fig_var.update_layout(template="plotly_white")
st.plotly_chart(fig_var, use_container_width=True)

# ========= CLUSTER KPI QUANTILES ==========
st.subheader("KPI Distribution per Cluster")

for cluster in sorted(filtered_df["Cluster"].unique()):
    count = (filtered_df["Cluster"] == cluster).sum()
    st.markdown(f"### Cluster {cluster}  \({count} companies)")

    # Extract rows for this cluster
    cluster_data = filtered_df[filtered_df["Cluster"] == cluster]
    q25 = cluster_data[KPI_FEATURES].quantile(0.25).rename("Bottom Quartile")
    q50 = cluster_data[KPI_FEATURES].median().rename("Median")
    q75 = cluster_data[KPI_FEATURES].quantile(0.75).rename("Top Quartile")

    # Combine
    summary = pd.concat([q25, q50, q75], axis=1).reset_index()
    summary.columns = ["KPI", "Bottom Quartile", "Median", "Top Quartile"]

    # Merge with metadata
    display_table = pd.merge(kpi_metadata, summary, how="inner", on="KPI")
    display_table = display_table[["KPI", "UoM", "What is better?", "Bottom Quartile", "Median", "Top Quartile"]]

    st.dataframe(display_table.style.format(precision=2), use_container_width=True)

