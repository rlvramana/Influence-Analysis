import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROC_DIR = BASE_DIR / "data" / "processed"
TABLE_DIR = BASE_DIR / "reports" / "tables"
GRAPH_DIR = BASE_DIR / "reports" / "graphs"

SCORES_PATH = PROC_DIR / "model_final_scores_all_campaigns.parquet"
SUMMARY_PATH = TABLE_DIR / "summary_top50_recommendations.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@st.cache_data
def load_scores():
    return pd.read_parquet(SCORES_PATH)


@st.cache_data
def load_summary():
    if SUMMARY_PATH.exists():
        return pd.read_csv(SUMMARY_PATH)
    return None


@st.cache_data
def load_top_table(campaign: str, top_k: int):
    csv_path = TABLE_DIR / f"{campaign}_top50_recommended_accounts.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df = df.sort_values("rank_by_model_score").head(top_k).reset_index(drop=True)
    return df


def load_graph_html(campaign: str, top_n: int = 60):
    html_path = GRAPH_DIR / f"{campaign}_network_top{top_n}_static.html"
    if not html_path.exists():
        return None
    return html_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Influence Analysis – SAGraph",
    layout="wide",
)

st.title("Influence Analysis dashboard")

st.markdown(
    """
This app shows:

- model performance by campaign
- recommended influencers per campaign
- an interactive network view of engagement around each campaign
"""
)

scores_all = load_scores()
campaigns = sorted(scores_all["product_name"].unique())
summary_df = load_summary()

# ------------------ Sidebar controls ------------------
st.sidebar.header("Controls")

campaign = st.sidebar.selectbox("Campaign", campaigns)

top_k = st.sidebar.slider(
    "How many of the top-50 recommendations to show in the table?",
    min_value=10,
    max_value=50,
    value=20,
    step=5,
)

account_view = st.sidebar.selectbox(
    "Which accounts to show?",
    ("All", "Official influencers only", "Non-official influencers only"),
)

# ------------------ Tabs ------------------
tab_overview, tab_recs, tab_graph = st.tabs(
    ["Overview", "Recommendations", "Network graph"]
)

# ------------------ Overview tab ------------------
with tab_overview:
    st.subheader("Model performance snapshot")

    if summary_df is not None:
        st.markdown("**Top-50 recommendation mix per campaign**")
        st.caption(
            "`recommended_total` is fixed at 50: our system always picks the top-50 "
            "accounts per campaign, then this table shows how many of those are "
            "official vs non-official."
        )
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("Summary CSV not found in reports/tables/ yet.")

    st.markdown(f"### Campaign: `{campaign}`")

    df_c = scores_all[scores_all["product_name"] == campaign].copy()
    n_accounts = len(df_c)
    n_high = int(df_c["label_high_engagement"].sum())

    c1, c2 = st.columns(2)
    c1.metric("Accounts in campaign", f"{n_accounts:,}")
    c2.metric("High-engagement accounts (label)", f"{n_high:,}")


# ------------------ Recommendations tab ------------------
with tab_recs:
    st.subheader(f"Top recommendations for campaign: `{campaign}`")

    df_top = load_top_table(campaign, top_k)
    if df_top is None:
        st.warning("Top-50 table not found for this campaign.")
    else:
        # Apply account type filter
        mask = pd.Series(True, index=df_top.index)
        if account_view == "Official influencers only":
            mask &= df_top["official_influencer"]
        elif account_view == "Non-official influencers only":
            mask &= ~df_top["official_influencer"]

        df_filtered = df_top[mask].reset_index(drop=True)

        st.caption(
            "Each row is an account in this campaign. "
            "`engagement_comments_plus_reposts` = comments + reposts on their ads. "
            "You are seeing the first N from our top-50 recommendation list, "
            "based on the slider in the sidebar."
        )

        st.dataframe(
            df_filtered,
            use_container_width=True,
            hide_index=True,
        )


# ------------------ Network graph tab ------------------
with tab_graph:
    st.subheader(f"Engagement network for campaign: `{campaign}`")
    st.caption(
        "Each circle is an account in this campaign. "
        "Size reflects graph centrality (PageRank). "
        "Red = top 10 recommended by the model, "
        "blue = official influencers, orange = other active accounts. "
        "Hover to see followers, engagement and model score."
    )

    html = load_graph_html(campaign, top_n=60)
    if html is None:
        st.warning("Network HTML file not found for this campaign.")
    else:
        components.html(html, height=750, scrolling=True)
