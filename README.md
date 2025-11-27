Influence Analysis on SAGraph

This project finds high-engagement influencers in social advertising campaigns using the SAGraph dataset (Weibo campaigns) and compares:
	1.	A simple baseline that ranks accounts by followers only, and
	2.	Models that use engagement + network structure (graph analytics).

The project is fully reproducible in Python and Streamlit.

Data source: SAGraph dataset and paper from the authors’ GitHub repository (public academic dataset, see: https://github.com/xiaoqzhwhu/SAGraph).
Modeling libraries: scikit-learn (logistic regression) and XGBoost (gradient boosting), using standard classification and evaluation methods from their official documentation (general machine learning practice).

⸻

1. Project structure

Influence-Analysis/
├── README.md
├── requirements.txt
├── app.py                        # Streamlit dashboard
├── data/
│   ├── raw/
│   │   └── SAGraph/
│   │       ├── product_info.jsonl
│   │       ├── abc_reading_profile.graph.anon
│   │       ├── abc_reading_interaction.graph.anon
│   │       ├── electric_toothbrush_profile.graph.anon
│   │       ├── electric_toothbrush_interaction.graph.anon
│   │       ├── intelligent_floor_scrubber_profile.graph.anon
│   │       ├── intelligent_floor_scrubber_interaction.graph.anon
│   │       ├── ruby_face_cream_profile.graph.anon
│   │       ├── ruby_face_cream_interaction.graph.anon
│   │       ├── spark_thinking_profile.graph.anon
│   │       ├── spark_thinking_interaction.graph.anon
│   │       ├── supor_boosted_showerhead_profile.graph.anon
│   │       └── supor_boosted_showerhead_interaction.graph.anon
│   ├── interim/                  # Per-product edges / intermediate features
│   └── processed/                # Final modeling tables
│       ├── features_per_product.parquet
│       ├── features_labeled.parquet
│       └── model_final_scores_all_campaigns.parquet
├── notebooks/
│   ├── 01_data_inventory_and_eda.ipynb
│   ├── 02_build_edges_and_network_features.ipynb
│   ├── 03_build_model_features_and_labels.ipynb
│   ├── 04_train_models_and_evaluate.ipynb
│   └── 05_build_network_graphs.ipynb
└── reports/
    ├── figures/                  # Static charts (optional)
    ├── tables/
    │   ├── summary_top50_recommendations.csv
    │   ├── abc_reading_top50_recommended_accounts.csv
    │   ├── electric_toothbrush_top50_recommended_accounts.csv
    │   ├── intelligent_floor_scrubber_top50_recommended_accounts.csv
    │   ├── ruby_face_cream_top50_recommended_accounts.csv
    │   ├── spark_thinking_top50_recommended_accounts.csv
    │   └── supor_boosted_showerhead_top50_recommended_accounts.csv
    └── graphs/
        ├── abc_reading_nodes.csv
        ├── abc_reading_edges.csv
        ├── abc_reading_network_top60_static.html
        ├── electric_toothbrush_nodes.csv
        ├── electric_toothbrush_edges.csv
        ├── electric_toothbrush_network_top60_static.html
        ├── ...
        └── supor_boosted_showerhead_network_top60_static.html

Notes:
	•	Folder names under notebooks/ can be different; what matters is the order (01 → 05).
	•	data/interim and reports/figures are created by the notebooks.

⸻

2. Python and libraries

Python version used: 3.11 (any recent 3.10+ should work).

2.1. Create environment

Using conda (recommended by scikit-learn and XGBoost docs for isolation; general best practice):

conda create -n influence-analysis python=3.11
conda activate influence-analysis

2.2. Install dependencies

requirements.txt should contain at least:

pandas
numpy
matplotlib
networkx
scikit-learn
xgboost
pyvis
streamlit
pyarrow      # for parquet read/write

Install:

pip install -r requirements.txt

These are standard, well-documented libraries for data analysis and modeling.

⸻

3. Download and place the SAGraph data
	1.	Go to the public SAGraph repository:
https://github.com/xiaoqzhwhu/SAGraph
	2.	Download the Weibo advertising data (the subset used here is the 6 products):
	•	product_info.jsonl
	•	*_profile.graph.anon
	•	*_interaction.graph.anon
	3.	Place all these files in:

data/raw/SAGraph/

Make sure the file names match the pattern in the structure above (for the 6 campaigns).

⸻

4. Reproducing the analysis (notebooks)

Run the notebooks in order. Each notebook saves outputs that the next notebook uses.

You can use VS Code, Jupyter Lab, or any notebook environment.
The steps below describe what each notebook does in plain language.

4.1. 01_data_inventory_and_eda.ipynb

Goal: understand the dataset and check that the files match.

Main steps:
	•	Read product_info.jsonl into a DataFrame.
	•	List all profile and interaction files and match them by product_name.
	•	For one campaign (e.g. abc_reading), preview:
	•	user profile stats (followers, friends),
	•	interaction counts (comments vs reposts),
	•	simple charts (e.g. interaction type counts).

Nothing is saved here; this is exploration to understand the data.

4.2. 02_build_edges_and_network_features.ipynb

Goal: for each product, build the user–user engagement network and compute network features.

Main steps:

For each product (campaign):
	1.	Load its *_interaction.graph.anon as a dict.
	2.	Build an edge list where each row is:
	•	src_user_id = audience account that interacts,
	•	dst_user_id = account that posted the ad (or ad-related content),
	•	interact_type = comment or reposts.
	3.	Remove self-loops (records where src_user_id == dst_user_id).
	4.	Create a directed NetworkX graph G from the edges.
	5.	Compute per-node (per account):
	•	in_degree = how many different accounts engaged with this account,
	•	out_degree = how many other accounts this account engaged with,
	•	pagerank = PageRank centrality on this network,
	•	kcore = core number on an undirected version of the graph.
	6.	Save per-campaign results to Parquet in data/interim/, for example:
	•	abc_reading_edges_clean.parquet
	•	abc_reading_net_features_clean.parquet.

These are used later when building modeling features.

4.3. 03_build_model_features_and_labels.ipynb

Goal: build one modeling table per user per campaign, then a combined table across all campaigns.

Main steps:

For each product:
	1.	Load:
	•	profile data (*_profile.graph.anon),
	•	aggregate engagement counts per user from the edge list,
	•	network features from data/interim.
	2.	Create per-user features:
	•	comment = number of comment interactions received,
	•	reposts = number of repost interactions received,
	•	total_engagement = comment + reposts,
	•	engagement_per_follower = total_engagement / max(1, followers),
	•	user_followers, user_friends,
	•	in_degree, out_degree, pagerank, kcore.
	3.	Apply log-transform where needed (standard practice for heavy-tailed counts):
	•	log1p_user_followers, log1p_user_friends,
	•	log1p_comment, log1p_reposts,
	•	log1p_total_engagement, log1p_in_degree, log1p_out_degree, log1p_pagerank.
	4.	Create a label for high engagement within each campaign:
	•	restrict to accounts that have at least one interaction,
	•	compute percentile of total_engagement within each product,
	•	mark the top ~20–25% as label_high_engagement = 1, the rest as 0.
(Exact threshold is based on quantiles; this is a modeling choice, not a fixed fact.)
	5.	Mark whether the account is in the official influencer list from product_info.jsonl:
	•	is_official_influencer = 1 if the user_id appears in that product’s influencer_ids, otherwise 0.
	6.	Save:
	•	data/processed/features_per_product.parquet
(all users with their features per product)
	•	data/processed/features_labeled.parquet
(only accounts with the high-engagement label, used for training).

4.4. 04_train_models_and_evaluate.ipynb

Goal: compare ranking methods and produce per-campaign recommendation tables.

Main steps (for each campaign):
	1.	Filter features_labeled.parquet to that product_name.
	2.	Split into training and test sets (e.g. 70/30), stratified by label_high_engagement.
	3.	Define feature sets:
	•	Baseline: log1p_user_followers, log1p_user_friends.
	•	Full: baseline + network features:
log1p_in_degree, log1p_out_degree, log1p_pagerank, kcore.
	4.	Train three models:
	•	logistic_baseline (logistic regression, baseline features),
	•	logistic_full (logistic regression, full features),
	•	xgboost_full (XGBoost classifier, full features).
	5.	Evaluate on the test set using:
	•	F1 score,
	•	ROC–AUC,
	•	precision@K (top-50 and top-100).
These are standard ranking metrics to check whether the model is good at putting high-engagement accounts near the top.
	6.	Train the chosen model on the full labeled data for that campaign (no train/test split) and score all candidate accounts in features_per_product for that campaign.
	7.	Save:
	•	data/processed/model_final_scores_all_campaigns.parquet
(all scores for all campaigns; includes score_logistic_full_final or similar).
	•	reports/tables/summary_top50_recommendations.csv
(per campaign: how many of the top-50 accounts are official vs non-official).
	•	reports/tables/{campaign}_top50_recommended_accounts.csv
(top-50 accounts per campaign with user_id, followers, engagement, network metrics, official flag, and final model score).

4.5. 05_build_network_graphs.ipynb

Goal: build interactive network graphs showing how recommended accounts sit in the engagement network.

Main steps for each campaign:
	1.	Load:
	•	cleaned edge list from data/interim,
	•	per-campaign features from features_per_product.parquet,
	•	final model scores from model_final_scores_all_campaigns.parquet.
	2.	Select a manageable subset of nodes to draw (for example, top 60 by in_degree or engagement).
	3.	Build a pyvis.Network graph:
	•	nodes:
	•	size based on PageRank or in_degree,
	•	color:
	•	red = top 10 model recommendations,
	•	blue = official influencers,
	•	orange = other active accounts.
	•	edges:
	•	drawn between accounts that interact around ads.
	4.	Add hover tooltips per node with:
	•	user id,
	•	followers,
	•	total engagement,
	•	in-degree,
	•	PageRank,
	•	model score,
	•	whether they are official influencers.
	5.	Save per campaign:
	•	{campaign}_nodes.csv and {campaign}_edges.csv in reports/graphs/,
	•	{campaign}_network_top60_static.html in reports/graphs/.

These HTML files are loaded by the Streamlit app.

⸻

5. Streamlit dashboard

The dashboard is defined in app.py.

It:
	•	loads model_final_scores_all_campaigns.parquet,
	•	loads summary_top50_recommendations.csv,
	•	loads {campaign}_top50_recommended_accounts.csv per campaign,
	•	embeds the {campaign}_network_top60_static.html graphs.

5.1. Running the app

From the project root:

conda activate influence-analysis
streamlit run app.py

A browser window will open (or you will get a local URL like http://localhost:8501).

5.2. What the app shows
	•	Sidebar
	•	Campaign selector (one of the six products).
	•	Slider: “How many of the top-50 recommendations to show in the table?”
	•	Dropdown: show All / Official only / Non-official only.
	•	Overview tab
	•	Table with recommended_total (always 50), recommended_official, recommended_non_official for each campaign (from summary_top50_recommendations.csv).
	•	Simple metrics:
	•	number of accounts in the current campaign,
	•	number of high-engagement accounts (label).
	•	Recommendations tab
	•	Interactive table that lists the first N accounts from the per-campaign top-50 recommendation list, filtered by account type.
	•	Columns include:
	•	user_id, official_influencer, user_followers,
	•	engagement counts, network metrics, and final model score.
	•	Network graph tab
	•	Embedded PyVis HTML graph for the current campaign.
	•	Interactive:
	•	hover to see tooltips,
	•	zoom and pan,
	•	see red (model’s top 10 picks), blue (officials), orange (others).

⸻

6. Repro steps summary
	1.	Clone the repository or copy this folder structure.
	2.	Create and activate the conda environment:

conda create -n influence-analysis python=3.11
conda activate influence-analysis
pip install -r requirements.txt


	3.	Download SAGraph data and place files under data/raw/SAGraph/.
	4.	Run notebooks in order (01 → 05) to generate:
	•	data/interim/*,
	•	data/processed/*,
	•	reports/tables/*,
	•	reports/graphs/*.
	5.	Start the Streamlit app:

streamlit run app.py


	6.	Use the browser UI to explore:
	•	cross-campaign summary,
	•	per-campaign recommended influencers,
	•	network graphs showing how they sit in the engagement network.
