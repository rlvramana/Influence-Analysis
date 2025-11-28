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
PRODUCT_FEATURES_PATH = PROC_DIR / "features_per_product.parquet"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@st.cache_data
def load_scores() -> pd.DataFrame:
    """Per-account features + final model scores for all campaigns."""
    return pd.read_parquet(SCORES_PATH)


@st.cache_data
def load_summary() -> pd.DataFrame | None:
    """
    Top-50 per campaign summary produced from the modelling notebook.
    Columns expected:
      product_name, recommended_total, recommended_official, recommended_non_official
    """
    if not SUMMARY_PATH.exists():
        return None
    df = pd.read_csv(SUMMARY_PATH)
    return df


@st.cache_data
def load_product_features() -> pd.DataFrame | None:
    """
    Per-campaign activity summary.
    We expect something like:
      product / product_name, users, edges, ...
    """
    if not PRODUCT_FEATURES_PATH.exists():
        return None

    df = pd.read_parquet(PRODUCT_FEATURES_PATH)
    # Normalise column names
    if "product" in df.columns and "product_name" not in df.columns:
        df = df.rename(columns={"product": "product_name"})
    return df


@st.cache_data
def load_top_table(campaign: str, top_k: int) -> pd.DataFrame | None:
    """
    Load the per-campaign top-50 table and return the top_k rows
    sorted by model rank.
    """
    csv_path = TABLE_DIR / f"{campaign}_top50_recommended_accounts.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df = df.sort_values("rank_by_model_score").head(top_k).reset_index(drop=True)
    return df


@st.cache_data
def load_graph_html(campaign: str, top_n: int = 60) -> str | None:
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

st.title("Influence analysis dashboard")

st.markdown(
    """
This app helps answer three questions for each campaign:

- How many people were involved and how many interactions? 
- How are these accounts connected in the conversation network?
- Which accounts do we recommend the brand should work with?  
"""
)

scores_all = load_scores()
campaigns = sorted(scores_all["product_name"].unique())

summary_df = load_summary()
product_features_df = load_product_features()

# ---------------------------------------------------------------------
# Pre-compute per-campaign aggregates for reuse
# ---------------------------------------------------------------------

# Active accounts (unique handles)
active_counts = (
    scores_all.groupby("product_name")["user_id"]
    .nunique()
    .reset_index(name="active_accounts_overall")
)

# Official influencers overall (not just in top-50)
official_overall = (
    scores_all[scores_all["is_official_influencer"] == 1]
    .groupby("product_name")["user_id"]
    .nunique()
    .reset_index(name="official_influencers_overall")
)

# Fallback interactions from account-level total_engagement if needed
if "total_engagement" in scores_all.columns:
    interactions_fallback = (
        scores_all.groupby("product_name")["total_engagement"]
        .sum()
        .reset_index(name="total_interactions")
    )
else:
    interactions_fallback = pd.DataFrame(
        {"product_name": scores_all["product_name"].unique(), "total_interactions": 0}
    )

# If we have product-level interaction counts, keep only the pieces we need.
# Otherwise, fall back to aggregating total_engagement.
if product_features_df is not None and "edges" in product_features_df.columns:
    interactions = product_features_df[["product_name", "edges"]].rename(
        columns={"edges": "total_interactions"}
    )
else:
    interactions = interactions_fallback.copy()

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


# ------------------ Tabs ------------------
tab_overview, tab_graph, tab_recs = st.tabs(
    ["Overview", "Conversation network", "Recommended accounts"]
)

# ---------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------
with tab_overview:
    st.subheader(f"Campaign overview: `{campaign}`")

    # ---- Headline metrics for this campaign ----
    row_active = active_counts[active_counts["product_name"] == campaign]
    row_official = official_overall[official_overall["product_name"] == campaign]
    row_interactions = interactions[interactions["product_name"] == campaign]

    active_val = (
        int(row_active["active_accounts_overall"].iloc[0])
        if not row_active.empty
        else 0
    )
    official_val = (
        int(row_official["official_influencers_overall"].iloc[0])
        if not row_official.empty
        else 0
    )
    other_creators_val = max(active_val - official_val, 0)
    total_interactions_val = (
        int(row_interactions["total_interactions"].iloc[0])
        if not row_interactions.empty
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("People who took part (unique accounts)", f"{active_val:,}")
    c2.metric("Official influencers in data", f"{official_val:,}")
    c3.metric("Other active creators", f"{other_creators_val:,}")
    c4.metric("Total interactions (comments + reposts)", f"{total_interactions_val:,}")

    st.caption(
        "“People who took part” counts any handle that posted about the product "
        "or interacted at least once (comment or repost) with a product-related post."
    )

    st.markdown("---")

    # ---- Model shortlist for this campaign (top-50) ----
    st.markdown("### Model shortlist for this campaign")

    if summary_df is not None:
        row_sum = summary_df[summary_df["product_name"] == campaign]
    else:
        row_sum = pd.DataFrame()

    if not row_sum.empty:
        rec_total = int(row_sum["recommended_total"].iloc[0])
        rec_official = int(row_sum["recommended_official"].iloc[0])
        rec_new = int(row_sum["recommended_non_official"].iloc[0])
    else:
        rec_total = rec_official = rec_new = 0

    s1, s2, s3 = st.columns(3)
    s1.metric("Accounts in shortlist", f"{rec_total}")
    s2.metric("Hired influencers in top-50", f"{rec_official}")
    s3.metric("New creators in top-50", f"{rec_new}")

    # Optional: show share of engagement generated by new creators in the shortlist
    top50_full = load_top_table(campaign, top_k=50)
    if (
        top50_full is not None
        and not top50_full.empty
        and "engagement_comments_plus_reposts" in top50_full.columns
    ):
        col_eng = "engagement_comments_plus_reposts"

        if "official_influencer" in top50_full.columns:
            # Treat anything non-1 as not official
            flag = top50_full["official_influencer"].fillna(0)
            total_eng = top50_full[col_eng].sum()
            eng_new = top50_full.loc[flag == 0, col_eng].sum()
            share_new = (eng_new / total_eng) if total_eng > 0 else 0.0
            pct_str = f"{share_new * 100:,.0f}%"
            st.markdown(
                f"In this shortlist, **new creators (not originally hired)** "
                f"generate roughly **{pct_str} of the engagement**."
            )

    st.markdown("---")

    # ---- Comparison table across campaigns (top-50 focused) ----
    st.markdown("### How this campaign compares to other campaigns")

    if summary_df is None:
        st.info("Summary CSV not found in reports/tables/ yet.")
    else:
        # Merge top-50 counts with overall activity
        cmp = summary_df.merge(active_counts, on="product_name", how="left").merge(
            interactions, on="product_name", how="left"
        )

        cmp_display = cmp.rename(
            columns={
                "product_name": "Campaign",
                "active_accounts_overall": "Active accounts (overall)",
                "total_interactions": "Total interactions (comments + reposts)",
                "recommended_official": "Hired influencers in top-50",
                "recommended_non_official": "New creators in top-50",
            }
        )

        # Clean up None / NaN
        if "Total interactions (comments + reposts)" in cmp_display.columns:
            cmp_display["Total interactions (comments + reposts)"] = (
                cmp_display["Total interactions (comments + reposts)"]
                .fillna(0)
                .astype(int)
            )

        cols_cmp = [
            "Campaign",
            "Active accounts (overall)",
            "Total interactions (comments + reposts)",
            "Hired influencers in top-50",
            "New creators in top-50",
        ]
        cmp_display = cmp_display[cols_cmp]

        st.dataframe(
            cmp_display.sort_values("Campaign"),
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------------------
# Network graph tab
# ---------------------------------------------------------------------
with tab_graph:
    st.subheader(f"Conversation network for: `{campaign}`")

    html = load_graph_html(campaign, top_n=60)
    if html is None:
        st.warning("Network HTML file not found for this campaign.")
    else:
        # Show the graph first
        components.html(html, height=750, scrolling=True)

        # Explanation below, collapsible
        with st.expander("How to read this network view"):
            st.markdown(
                """
This view shows how accounts are connected in the campaign’s conversation:

- Each circle is an account.  
- Larger circles have more followers.  
- Red = top 10 accounts from our recommendation list.  
- Blue = official influencers.  
- Orange = other active accounts.  

Hover over a circle to see followers and engagement (comments + reposts).
"""
            )

# ---------------------------------------------------------------------
# Recommended accounts tab
# ---------------------------------------------------------------------
with tab_recs:
    st.subheader(f"Recommended accounts for: `{campaign}`")

    df_top = load_top_table(campaign, top_k)
    if df_top is None or df_top.empty:
        st.warning("Top-50 table not found for this campaign.")
    else:
        # Show all accounts (no filtering)
        df_filtered = df_top.reset_index(drop=True)

        # Business-friendly view
        display_df = df_filtered.copy()

        # Clean Yes/No flags with safe handling of missing values
        display_df["official_influencer"] = (
            display_df["official_influencer"].fillna(0).astype(int)
        )

        display_df["Official influencer?"] = display_df["official_influencer"].map(
            {1: "Yes", 0: "No"}
        )

        # Select and rename columns for display (no model score column)
        cols_order = [
            "rank_by_model_score",
            "user_id",
            "Official influencer?",
            "followers",
            "engagement_comments_plus_reposts",
            "people_who_engaged",
            "pagerank_centrality",
        ]

        display_df = display_df[cols_order].rename(
            columns={
                "rank_by_model_score": "Rank in our recommendation list",
                "user_id": "Account ID",
                "followers": "Followers",
                "engagement_comments_plus_reposts": "Total engagement (comments + reposts)",
                "people_who_engaged": "Number of unique people who engaged",
                "pagerank_centrality": "Network influence score",
            }
        )

        # Show the table first
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # Column explanations below, collapsible
        with st.expander("What do these columns mean?"):
            st.markdown(
                """
Each row is an account in this campaign. Key columns:

- **Official influencer?** – whether this account was in the brand’s original influencer list.  
- **Followers** – how many followers this account has on the platform.  
- **Total engagement (comments + reposts)** – total number of comments and reposts on this account’s posts about this product.  
- **Number of unique people who engaged** – how many different accounts interacted with this account’s posts.  
- **Network influence score** – how central this account is in the campaign’s conversation network (higher means more central).  

Accounts are ranked by a model that combines engagement and network position to highlight the strongest potential promoters.
"""
            )
