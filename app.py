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
        df = pd.read_csv(SUMMARY_PATH)
        # Make summary column names business friendly
        df = df.rename(
            columns={
                "product_name": "Campaign",
                "recommended_total": "Top-50 accounts",
                "recommended_official": "Top-50 – official influencers",
                "recommended_non_official": "Top-50 – non-official creators",
            }
        )
        return df
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
This app helps answer three questions for each campaign:

- How many strong voices did we find?
- Who are the top accounts we recommend?
- How are these accounts connected in the conversation network?
"""
)

scores_all = load_scores()
campaigns = sorted(scores_all["product_name"].unique())
summary_df = load_summary()

# ------------------ Sidebar controls ------------------
st.sidebar.header("Controls")

campaign = st.sidebar.selectbox("Choose a campaign", campaigns)

top_k = st.sidebar.slider(
    "How many of the top-50 recommended accounts to show in the table?",
    min_value=10,
    max_value=50,
    value=20,
    step=5,
)

account_view = st.sidebar.selectbox(
    "Which accounts do you want to see?",
    ("All", "Official influencers only", "Non-official influencers only"),
)

# ------------------ Tabs ------------------
tab_overview, tab_recs, tab_graph = st.tabs(
    ["Overview", "Recommended accounts", "Conversation network"]
)

# ------------------ Overview tab ------------------
with tab_overview:
    st.subheader("Campaign overview")

    if summary_df is not None:
        st.markdown("**How many recommended accounts per campaign?**")
        st.markdown(
            """
For each campaign we always build a top-50 list.

- “Top-50 – official influencers” – how many of the 50 are from the brand’s original influencer list.  
- “Top-50 – non-official creators” – how many are additional people we discovered.
"""
        )
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("Summary CSV not found in reports/tables/ yet.")

    st.markdown(f"### Selected campaign: `{campaign}`")

    df_c = scores_all[scores_all["product_name"] == campaign].copy()
    n_accounts = len(df_c)

    c1 = st.columns(1)[0]
    c1.metric("Accounts in this campaign", f"{n_accounts:,}")


# ------------------ Recommendations tab ------------------
with tab_recs:
    st.subheader(f"Recommended accounts for: `{campaign}`")

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

        # Build a business-friendly view
        display_df = df_filtered.copy()

        # Clean Yes/No flags with safe handling of missing values
        display_df["official_influencer"] = (
            display_df["official_influencer"].fillna(False).astype(bool)
        )

        display_df["Official influencer?"] = display_df["official_influencer"].map(
            {True: "Yes", False: "No"}
        )

        # Select and rename columns for display (no top-20% column)
        cols_order = [
            "rank_by_model_score",
            "user_id",
            "Official influencer?",
            "followers",
            "engagement_comments_plus_reposts",
            "people_who_engaged",
            "pagerank_centrality",
            "model_score",
        ]

        display_df = display_df[cols_order].rename(
            columns={
                "rank_by_model_score": "Rank in our recommendation list",
                "user_id": "Account ID",
                "followers": "Followers",
                "engagement_comments_plus_reposts": "Total engagement (comments + reposts)",
                "people_who_engaged": "Number of unique people who engaged",
                "pagerank_centrality": "Network influence score",
                "model_score": "Model confidence (0–1)",
            }
        )

        # Show the table first
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # Then show the column explanations below, in a collapsible section
        with st.expander("What do these columns mean?"):
            st.markdown(
                """
Each row is an account in this campaign. Key columns:

- **Official influencer?** – whether this account was in the brand’s original influencer list.  
- **Total engagement (comments + reposts)** – total number of comments and reposts on this account’s posts about this product.  
- **Number of unique people who engaged** – how many different accounts interacted with this account’s posts.  
- **Network influence score** – how central this account is in the campaign’s conversation network (higher means more central).  
- **Model confidence (0–1)** – how strongly our model believes this account behaves like a high-engagement influencer.
"""
            )


# ------------------ Network graph tab ------------------
with tab_graph:
    st.subheader(f"Conversation network for: `{campaign}`")

    html = load_graph_html(campaign, top_n=60)
    if html is None:
        st.warning("Network HTML file not found for this campaign.")
    else:
        # Show the graph first
        components.html(html, height=750, scrolling=True)

        # Then show the explanation below, in a collapsible section
        with st.expander("How to read this network view"):
            st.markdown(
                """
This view shows how accounts are connected in the campaign’s conversation:

- Each circle is an account.  
- Larger circles have higher network influence (they are more central in the conversation graph).  
- Red = top 10 recommended accounts.  
- Blue = official influencers.  
- Orange = other active accounts.  

Hover over a circle to see followers, engagement, and model confidence.
"""
            )
