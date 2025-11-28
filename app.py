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
def load_scores() -> pd.DataFrame:
    return pd.read_parquet(SCORES_PATH)


@st.cache_data
def load_summary() -> pd.DataFrame | None:
    if SUMMARY_PATH.exists():
        return pd.read_csv(SUMMARY_PATH)
    return None


@st.cache_data
def load_top_table(campaign: str, top_k: int) -> pd.DataFrame | None:
    """
    Load the per-campaign top-50 recommendations table and make sure we have
    a consistent rank column called `rank_by_model_score`.
    """
    csv_path = TABLE_DIR / f"{campaign}_top50_recommended_accounts.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    # 1) Normalise the rank column name
    rank_col = None
    for cand in ["rank_by_model_score", "rank_by_score_final", "rank_by_score"]:
        if cand in df.columns:
            rank_col = cand
            break

    if rank_col is not None:
        df = df.sort_values(rank_col).reset_index(drop=True)
        if rank_col != "rank_by_model_score":
            df = df.rename(columns={rank_col: "rank_by_model_score"})
    else:
        # No rank column – create one based on a score column if possible
        score_col = None
        for cand in ["model_score", "score_logistic_full_final", "score_xgb_full"]:
            if cand in df.columns:
                score_col = cand
                break
        if score_col is not None:
            df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        df.insert(0, "rank_by_model_score", df.index + 1)

    # Restrict to the requested top_k rows
    df = df.head(top_k).reset_index(drop=True)
    return df


def load_graph_html(campaign: str, top_n: int = 60) -> str | None:
    html_path = GRAPH_DIR / f"{campaign}_network_top{top_n}_static.html"
    if not html_path.exists():
        return None
    return html_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Hidden Influencer Finder",
    layout="wide",
)

st.title("Hidden Influencer Finder")

st.markdown(
    """
This tool helps you answer one key question:

**Who is actually driving the conversation in each campaign – not just the people we hired?**

You can:
- see how many recommended accounts are brand-selected influencers vs organic creators,
- browse a ranked list of top influencer candidates,
- and explore an interactive map of how conversations flow between accounts.
"""
)

scores_all = load_scores()
campaigns = sorted(scores_all["product_name"].unique())
summary_df = load_summary()

tab_summary, tab_recs, tab_graph = st.tabs(
    ["Campaign summary", "Top influencer candidates", "Conversation map"]
)

# ------------------ Campaign summary tab ------------------
with tab_summary:
    st.subheader("Campaign summary")

    campaign_summary = st.selectbox(
        "Choose a campaign",
        campaigns,
        key="campaign_summary",
    )

    if summary_df is not None:
        display_summary = summary_df.rename(
            columns={
                "product_name": "Campaign",
                "recommended_total": "Total accounts in recommendation list",
                "recommended_official": "Brand-selected influencers in list",
                "recommended_non_official": "Organic creators in list",
            }
        )

        st.markdown("**Top-50 recommendation mix across campaigns**")
        st.markdown(
            "For each campaign, we always build a **top-50 recommendation list**. "
            "This table shows how many of those 50 are brand-selected influencers "
            "versus organic creators who were not hired."
        )
        st.dataframe(display_summary, use_container_width=True)
    else:
        st.info("Summary CSV not found in reports/tables/ yet.")

    st.markdown(f"### Campaign details: `{campaign_summary}`")

    df_c = scores_all[scores_all["product_name"] == campaign_summary].copy()
    n_accounts = len(df_c)
    n_high = int(df_c["label_high_engagement"].sum())

    col1, col2 = st.columns(2)
    col1.metric("Accounts in this campaign", f"{n_accounts:,}")
    col2.metric("High-engagement accounts (top 20%)", f"{n_high:,}")

    st.markdown(
        "_High-engagement accounts are those in the top ~20% of this campaign "
        "by engagement quality (how much they were able to spark reactions)._"
    )


# ------------------ Top influencer candidates tab ------------------
with tab_recs:
    st.subheader("Top influencer candidates")

    campaign_recs = st.selectbox(
        "Choose a campaign",
        campaigns,
        key="campaign_recs",
    )

    top_k = st.slider(
        "How many of the top-50 recommended accounts would you like to see in the list?",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
    )

    account_view = st.selectbox(
        "Which accounts should be shown in the list?",
        (
            "All accounts",
            "Brand-selected influencers only",
            "Organic creators (not hired) only",
        ),
        key="account_view_recs",
    )

    df_top = load_top_table(campaign_recs, top_k)

    if df_top is None:
        st.warning("Top-50 recommendation table not found for this campaign.")
    else:
        # Rename technical columns into business language
        rename_map_raw = {}

        if "user_id" in df_top.columns:
            rename_map_raw["user_id"] = "Account ID"
        if "is_official_influencer" in df_top.columns:
            rename_map_raw["is_official_influencer"] = "Brand-selected influencer?"
        if "official_influencer" in df_top.columns:
            rename_map_raw["official_influencer"] = "Brand-selected influencer?"
        if "user_followers" in df_top.columns:
            rename_map_raw["user_followers"] = "Followers"
        if "followers" in df_top.columns:
            rename_map_raw["followers"] = "Followers"
        if "total_engagement" in df_top.columns:
            rename_map_raw["total_engagement"] = (
                "Engagement on this campaign (comments + reposts)"
            )
        if "engagement_comments_plus_reposts" in df_top.columns:
            rename_map_raw["engagement_comments_plus_reposts"] = (
                "Engagement on this campaign (comments + reposts)"
            )
        if "in_degree" in df_top.columns:
            rename_map_raw["in_degree"] = "People who engaged with this account"
        if "people_who_engaged" in df_top.columns:
            rename_map_raw["people_who_engaged"] = (
                "People who engaged with this account"
            )
        if "pagerank" in df_top.columns:
            rename_map_raw["pagerank"] = "Network influence score"
        if "pagerank_centrality" in df_top.columns:
            rename_map_raw["pagerank_centrality"] = "Network influence score"
        if "model_score" in df_top.columns:
            rename_map_raw["model_score"] = "Recommendation score"
        if "score_logistic_full_final" in df_top.columns:
            rename_map_raw["score_logistic_full_final"] = "Recommendation score"
        if "label_high_engagement" in df_top.columns:
            rename_map_raw["label_high_engagement"] = (
                "High-engagement account (top 20%)?"
            )

        rename_map_raw["rank_by_model_score"] = "Rank in recommendation list"

        df_renamed = df_top.rename(columns=rename_map_raw)

        # Turn flags into Yes / No
        flag_col = "Brand-selected influencer?"
        if flag_col in df_renamed.columns:
            df_renamed[flag_col] = (
                df_renamed[flag_col].map({1: "Yes", 0: "No"}).fillna("No")
            )

        high_col = "High-engagement account (top 20%)?"
        if high_col in df_renamed.columns:
            df_renamed[high_col] = (
                df_renamed[high_col].map({1: "Yes", 0: "No"}).fillna("No")
            )

        # Filter by account type using Yes/No
        if flag_col in df_renamed.columns:
            mask = pd.Series(True, index=df_renamed.index)
            if account_view == "Brand-selected influencers only":
                mask &= df_renamed[flag_col] == "Yes"
            elif account_view == "Organic creators (not hired) only":
                mask &= df_renamed[flag_col] == "No"
            df_filtered = df_renamed[mask].reset_index(drop=True)
        else:
            df_filtered = df_renamed

        st.markdown(
            "**How to read this list**  \n"
            "Each row is an account that talked about this product. "
            "The list is ordered by our **recommendation score**, which combines "
            "account size with their position in the conversation network."
        )

        with st.expander("What do these columns mean?"):
            st.markdown(
                """
- **Rank in recommendation list** – Position in our top-50 list for this campaign.
- **Brand-selected influencer?** – “Yes” if the brand or platform already marked this account as an official influencer for the product.
- **Followers** – Number of people following this account (audience size).
- **Engagement on this campaign (comments + reposts)** – How many times people commented on or reposted this account’s sponsored content for this product.
- **People who engaged with this account** – Number of distinct accounts that interacted with this account’s sponsored posts.
- **Network influence score** – How central the account is in the campaign’s conversation network. Higher values mean “more people, and more important people, interact with them”.
- **Recommendation score** – A number between 0 and 1 from our model. The closer to 1, the stronger we think this account is as a potential influencer for this product.
- **High-engagement account (top 20%)?** – “Yes” if this account is in the top 20% by engagement quality within this campaign.
"""
            )

        cols_order = [
            "Rank in recommendation list",
            "Account ID",
            "Brand-selected influencer?",
            "Followers",
            "Engagement on this campaign (comments + reposts)",
            "People who engaged with this account",
            "Network influence score",
            "Recommendation score",
            "High-engagement account (top 20%)?",
        ]
        cols_to_show = [c for c in cols_order if c in df_filtered.columns]

        st.dataframe(
            df_filtered[cols_to_show],
            use_container_width=True,
            hide_index=True,
        )


# ------------------ Conversation map tab ------------------
with tab_graph:
    st.subheader("Conversation map")

    campaign_graph = st.selectbox(
        "Choose a campaign",
        campaigns,
        key="campaign_graph",
    )

    st.markdown(
        """
This map shows how product-related conversations flow between accounts
for the selected campaign.

- Each circle is an account talking about the product.  
- Bigger circles represent accounts with **more influence** in this campaign (more unique people interacting with them and their neighbours).  
- Red circles are the **top-10 recommended accounts**.  
- Blue circles are **brand-selected influencers**.  
- Orange circles are **organic creators** (not hired).

Hover over a circle to see followers, engagement and recommendation score.
"""
    )

    with st.expander("What do size, colour and lines mean?"):
        st.markdown(
            """
- **Circle size** – Based mainly on the network influence score. Larger circles are more central in the product’s discussion and get attention from many people and other influential accounts.
- **Circle colour**  
  - Red – top-10 accounts recommended by our model for this campaign  
  - Blue – brand-selected influencers  
  - Orange – organic creators (not officially hired)
- **Lines between circles** – At least one interaction happened between the two accounts around the product’s ads (comment or repost). Thicker lines mean more interactions.
"""
        )

    html = load_graph_html(campaign_graph, top_n=60)
    if html is None:
        st.warning("Network HTML file not found for this campaign.")
    else:
        components.html(html, height=750, scrolling=True)
