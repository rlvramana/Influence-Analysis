# Influence Analysis on SAGraph

This project uses the **SAGraph** dataset (social advertising campaigns on Weibo) to identify **high-impact influencers** based on:

- how much engagement they receive (comments + reposts), and  
- how central they are in the engagement network (network features like in-degree, PageRank, k-core),  
- not just their follower count.

The end result is:

- a ranked list of recommended influencers per campaign (official and non-official),  
- comparison of “followers only” vs “followers + network” models,  
- interactive network graphs per campaign (PyVis),  
- an interactive **Streamlit dashboard** to explore everything.

---

## 1. Project structure

The repo uses a fairly standard data-science layout:

```text
Influence-Analysis/
├── app.py                     # Streamlit app
├── requirements.txt           # Python dependencies
├── README.md                  # This file

├── data/
│   ├── raw/
│   │   └── SAGraph/           # Original SAGraph JSON/JSONL files
│   ├── interim/               # Cleaned edges, network features per product
│   └── processed/             # Modeling tables, labeled data, model scores

├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_build_features.ipynb
│   ├── 03_model_per_campaign.ipynb
│   ├── 04_prepare_recommendations.ipynb
│   └── 05_network_graphs.ipynb

├── reports/
│   ├── tables/                # CSVs for tables used in Streamlit
│   └── graphs/                # HTML network graphs (PyVis) per campaign

└── src/                       # (optional) helper functions if you add them

# Influence Analysis on SAGraph

This repository contains an end-to-end, reproducible project that uses the
[SAGraph](https://github.com/xiaoqzhwhu/SAGraph) dataset to identify
high-engagement influencers in advertising campaigns using network
analytics, not just follower counts.

The workflow is:

1. Load and explore the raw SAGraph data.
2. Build user–user interaction graphs for each product (campaign).
3. Engineer profile, engagement and network features per account.
4. Define a high-engagement label inside each campaign.
5. Train and compare simple vs network-aware models.
6. Generate recommendations and interactive network visualisations.
7. Expose everything in a small Streamlit dashboard.

The code is organised so that you can re-run the full pipeline from scratch.

---

## 1. Environment and installation

### 1.1. Prerequisites

- Python 3.10 or 3.11  
- Git  
- A virtual environment tool (conda, venv, etc.)

All Python dependencies are listed in `requirements.txt`.

### 1.2. Clone and set up

```bash
# Clone this repo
git clone <your-repo-url>.git
cd Influence-Analysis

# (Optional but recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # on macOS / Linux
# .venv\Scripts\activate           # on Windows PowerShell

# Install dependencies
pip install -r requirements.txt

If you use conda, the steps are similar:
conda create -n influence-analysis python=3.11
conda activate influence-analysis
pip install -r requirements.txt

2. Data layout

This project expects the SAGraph data to be available under
data/raw/SAGraph in the following structure:
data/
  raw/
    SAGraph/
      product_info.jsonl
      abc_reading_profile.graph.anon
      abc_reading_interaction.graph.anon
      electric_toothbrush_profile.graph.anon
      electric_toothbrush_interaction.graph.anon
      intelligent_floor_scrubber_profile.graph.anon
      intelligent_floor_scrubber_interaction.graph.anon
      ruby_face_cream_profile.graph.anon
      ruby_face_cream_interaction.graph.anon
      spark_thinking_profile.graph.anon
      spark_thinking_interaction.graph.anon
      supor_boosted_showerhead_profile.graph.anon
      supor_boosted_showerhead_interaction.graph.anon
  interim/     # created by notebooks (edge lists, per-product graphs)
  processed/   # created by notebooks (features, labels, model outputs)

  You can obtain the raw files from the official SAGraph repository and copy
them into data/raw/SAGraph.

The notebooks never modify the raw files; all derived data is
written into data/interim and data/processed.

Influence-Analysis/
│
├── data/
│   ├── raw/            # raw SAGraph data (you add this)
│   ├── interim/        # intermediate edge lists and features (generated)
│   └── processed/      # modeling tables and final features (generated)
│
├── lib/                # JS libraries for vis-network (used by notebook 04)
│   ├── bindings/
│   ├── tom-select/
│   └── vis-9.1.2/
│
├── notebooks/
│   ├── 01_explore_and_edges.ipynb
│   ├── 02_features_labeling.ipynb
│   ├── 03_modeling_and_recommendations.ipynb
│   └── 04_network_visualisation.ipynb
│
├── reports/
│   ├── graphs/         # HTML network graphs and graph CSVs (generated)
│   └── tables/         # CSV tables used by Streamlit (generated)
│
├── scripts/            # (optional) helper scripts, not required to run
│
├── src/
│   └── __init__.py     # placeholder so src is a package
│
├── app.py              # Streamlit dashboard
├── requirements.txt
└── README.md

4. Step-by-step pipeline

All the heavy lifting is done in four notebooks, which are meant to be
run in order.

You can run them from VS Code or Jupyter. Each notebook is self-contained
and prints out what files it has created.

4.1. Notebook 01 – explore data and build edge lists

notebooks/01_explore_and_edges.ipynb

What it does:
	•	Loads product_info.jsonl and checks basic campaign statistics.
	•	For each *_profile.graph.anon file:
	•	builds a profiles table with user id, followers, friends, etc.
	•	For each *_interaction.graph.anon file:
	•	converts it into an edge list:
	•	one row per interaction,
	•	src_user_id = audience account,
	•	dst_user_id = account whose ad was engaged with,
	•	interact_type = “comment” or “reposts”.
	•	Drops self-loops and writes cleaned per-product edge lists to
data/interim/<product>_edges_clean.parquet.
	•	Saves per-product profile tables to data/interim/<product>_profiles.parquet.

Outputs (per product):
	•	data/interim/<product>_edges_clean.parquet
	•	data/interim/<product>_profiles.parquet

4.2. Notebook 02 – features and labeling

notebooks/02_features_labeling.ipynb

What it does:
	•	Reads the per-product edge lists and profiles created in notebook 01.
	•	Aggregates per-user engagement counts:
	•	number of comments,
	•	number of reposts,
	•	total engagement = comments + reposts.
	•	Adds profile information:
	•	followers,
	•	friends,
	•	engagement per follower, etc.
	•	Builds a user–user NetworkX graph for each product and computes network
features:
	•	in-degree / out-degree,
	•	PageRank,
	•	k-core number.
	•	Applies log transforms (e.g. log1p_user_followers) to stabilise very
skewed distributions.
	•	Defines a high-engagement label inside each campaign:
	•	for each product, takes the top ~20% of accounts by engagement-per-
follower as label 1, the rest as label 0.
	•	Merges everything into a single modeling table.

Outputs:
	•	data/processed/features_per_product.parquet – per-product features.
	•	data/processed/features_labeled.parquet – main modeling table with
one row per (product, user) and a label_high_engagement column.

4.3. Notebook 03 – modeling and recommendations

notebooks/03_modeling_and_recommendations.ipynb

What it does:
	•	Loads features_labeled.parquet.
	•	For each campaign (product_name):
	1.	Splits accounts into train/test (stratified by the label).
	2.	Trains three models:
	•	logistic_baseline – uses only profile size:
log1p_user_followers, log1p_user_friends.
	•	logistic_full – uses profile + network features (no raw
engagement counts to avoid leakage).
	•	xgboost_full – same feature set as logistic_full, but a
non-linear gradient boosted tree model.
	3.	Evaluates each model on the test set with:
	•	F1 score,
	•	ROC-AUC,
	•	precision@50 and precision@100
(share of truly high-engagement accounts in the top 50 / 100
recommended accounts).
	4.	For each campaign, produces a top-50 recommendation list based on
the best model (logistic_full), with a flag showing whether each
account is an official influencer or not.
	•	Aggregates metrics across all campaigns into a single table.

Outputs:
	•	data/processed/model_final_scores_all_campaigns.parquet – model scores
and labels for all (product, user) pairs.
	•	reports/tables/<product>_top50_recommended_accounts.csv – top-50
accounts per campaign, with features and scores used by the dashboard.
	•	reports/tables/summary_top50_recommendations.csv – one row per
campaign, showing how many of the top-50 are official vs non-official.

4.4. Notebook 04 – network visualisation

notebooks/04_network_visualisation.ipynb

What it does:
	•	Reads per-product edge lists and the final recommendation table.
	•	For each campaign, builds a smaller sub-graph focused on the most
engaged accounts (by in-degree and model score).
	•	Uses vis-network (via a small HTML/JS template in lib/) to create an
interactive graph:
	•	each node is an account,
	•	node size reflects PageRank (structural influence),
	•	node colour reflects role:
	•	red = top 10 recommended by the model,
	•	blue = official influencers from product_info.jsonl,
	•	orange = other engaged accounts,
	•	edges represent interactions (comments + reposts),
	•	tooltips show followers, engagement and model score.

Outputs:
	•	reports/graphs/<product>_network_top60_static.html – interactive HTML
network graph per campaign.
	•	reports/graphs/<product>_nodes.csv and
reports/graphs/<product>_edges.csv – underlying node/edge tables.

⸻

5. Streamlit dashboard

app.py is a small Streamlit app that lets you browse results without
opening notebooks.

5.1. Running the app

From the project root:
streamlit run app.py
Streamlit will open a browser window (or give you a local URL) with three
tabs:
	1.	Overview
	•	Shows, for each campaign, how many of the top-50 recommended
accounts are official vs non-official.
	•	For the currently selected campaign, shows how many accounts we have
and how many are labelled high-engagement.
	2.	Recommendations
	•	Lets you pick a campaign and how many of the top-50 accounts to
display (for example, top 10, top 20, …).
	•	You can filter to:
	•	All accounts,
	•	Only official influencers,
	•	Only non-official accounts.
	•	The table includes follower counts, engagement, network metrics and
the model score used for ranking.
	3.	Network graph
	•	Shows the interactive network graph for the selected campaign
(generated in notebook 04).
	•	You can pan/zoom and hover nodes to see followers, engagement and
model score.

The app reads everything from the data/processed and reports/ folders,
so make sure you have run notebooks 01–04 first.

⸻

6. Reproducing the full pipeline (checklist)
	1.	Download data
	•	Clone this repo.
	•	Download the SAGraph dataset from the official source.
	•	Place the files under data/raw/SAGraph/ as described above.
	2.	Create environment & install packages
	•	Create a virtual environment (optional but recommended).
	•	pip install -r requirements.txt.
	3.	Run notebooks in order
	•	01_explore_and_edges.ipynb
	•	02_features_labeling.ipynb
	•	03_modeling_and_recommendations.ipynb
	•	04_network_visualisation.ipynb
	4.	Launch the dashboard
	•	streamlit run app.py

After these steps you should be able to:
	•	See model performance across campaigns,
	•	Browse recommended influencers per campaign,
	•	Explore the underlying engagement networks interactively.

⸻

7. References
	•	SAGraph dataset and paper – official repository:
https://github.com/xiaoqzhwhu/SAGraph
	•	Network analysis: NetworkX – https://networkx.org/documentation/stable/
	•	Modeling:
	•	scikit-learn – https://scikit-learn.org/stable/
	•	XGBoost – https://xgboost.readthedocs.io/en/stable/
	•	Dashboard: Streamlit – https://docs.streamlit.io/
	•	Network visualisation: vis-network – https://visjs.github.io/vis-network/