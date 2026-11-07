from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data" / "processed"
# Small product lookup committed to the repo (made by scripts/make_product_mapping.py).
# Falls back to the large cleaned file only when running locally.
MAPPING_FILE = ARTIFACTS / "association" / "product_mapping.csv"
if not MAPPING_FILE.exists():
    MAPPING_FILE = DATA / "cleaned_transactions.csv"

st.set_page_config(
    page_title="RetailPulse | Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_product_mapping(path: str) -> dict[str, str]:
    if not Path(path).exists():
        st.error("Product lookup file is missing. Run `python scripts/make_product_mapping.py` and commit "
                 "`artifacts/association/product_mapping.csv`.")
        st.stop()
    df = pd.read_csv(path, usecols=["StockCode", "Description"])
    df = df.dropna(subset=["StockCode", "Description"])
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    return (df.drop_duplicates("StockCode", keep="first")
              .set_index("StockCode")["Description"].to_dict())


def read_artifact(relative: str, **kwargs) -> pd.DataFrame:
    path = ROOT / relative
    if not path.exists():
        st.error(f"Required project artifact is missing: `{relative}`")
        st.stop()
    return load_csv(str(path), **kwargs)


def read_metadata(relative: str) -> dict:
    path = ROOT / relative
    if not path.exists():
        return {}
    return load_json(str(path))


def pretty_product(code: str, mapping: dict[str, str]) -> str:
    code = str(code).strip()
    description = mapping.get(code)
    return f"{description} ({code})" if description else code


def as_codes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


# ---------- Sidebar ----------
st.sidebar.title("🛍️ RetailPulse")
st.sidebar.caption("Retail customer intelligence platform")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Customer Segmentation", "Inactivity Prediction",
     "Product Associations", "Recommendations", "About & Methodology"],
)
st.sidebar.divider()
st.sidebar.caption("Built from the project's saved model artifacts and evaluation outputs.")

# ---------- Overview ----------
if page == "Overview":
    st.title("RetailPulse Dashboard")
    st.write("A unified view of customer segments, inactivity risk, product associations, and basket recommendations.")

    segments = read_artifact("artifacts/segmentation/customer_segments.csv")
    scores = read_artifact("artifacts/inactivity/inactivity_v2_customer_scores.csv")
    assoc_meta = read_metadata("artifacts/association/association_run_summary.json")
    rec_eval = read_artifact("artifacts/recommendation/recommendation_evaluation.csv")
    rec_meta = read_metadata("artifacts/recommendation/recommendation_engine_metadata.json")

    latest_snapshot = pd.to_datetime(scores["Snapshot"], errors="coerce").max()
    latest_scores = scores[pd.to_datetime(scores["Snapshot"], errors="coerce") == latest_snapshot]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segmented customers", f"{segments['Customer ID'].nunique():,}")
    c2.metric("Latest risk-score records", f"{len(latest_scores):,}")
    c3.metric("Invoice baskets mined", f"{int(assoc_meta.get('invoice_baskets', 0)):,}")
    c4.metric("Association rules", f"{int(assoc_meta.get('filtered_association_rules', 0)):,}")

    st.subheader("Customer segmentation")
    segment_counts = segments["Segment"].value_counts().sort_index()
    left, right = st.columns(2)
    with left:
        st.bar_chart(segment_counts.rename("Customers"))
    with right:
        profiles = read_artifact("artifacts/segmentation/segment_profiles.csv")
        st.dataframe(profiles, width="stretch", hide_index=True)

    st.subheader("Recommendation evaluation")
    st.caption("Offline temporal holdout results from the saved evaluation; these are not live business outcomes.")
    st.dataframe(rec_eval, width="stretch", hide_index=True)

    st.info("Inactivity scores are shown for the latest snapshot available in the saved score file: "
            f"{latest_snapshot.date() if pd.notna(latest_snapshot) else 'date unavailable'}. "
            "They are model scores, not guarantees that a customer will become inactive.")

# ---------- Segmentation ----------
elif page == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.write("Explore the saved K-means customer assignments and RFM-style segment profiles.")
    customers = read_artifact("artifacts/segmentation/customer_segments.csv")
    profiles = read_artifact("artifacts/segmentation/segment_profiles.csv")

    a, b = st.columns([1, 2])
    with a:
        chosen = st.selectbox("Select segment", ["All"] + sorted(customers["Segment"].dropna().unique().tolist()))
    view = customers if chosen == "All" else customers[customers["Segment"] == chosen]
    x, y, z = st.columns(3)
    x.metric("Customers in view", f"{view['Customer ID'].nunique():,}")
    y.metric("Average frequency", f"{view['Frequency'].mean():.2f}")
    z.metric("Average monetary value", f"{view['Monetary'].mean():,.2f}")

    st.subheader("Segment profiles")
    st.dataframe(profiles, width="stretch", hide_index=True)
    diagnostics_path = ROOT / "artifacts" / "research_improvements" / "segmentation_cluster_count_comparison_full.csv"
    stability_path = ROOT / "artifacts" / "research_improvements" / "segmentation_temporal_stability.csv"
    if diagnostics_path.exists():
        with st.expander("Research validation: cluster-count selection and temporal stability"):
            st.dataframe(load_csv(str(diagnostics_path)), width="stretch", hide_index=True)
            if stability_path.exists():
                st.caption("Temporal ARI compares independently fitted clusters on two non-overlapping 12-month windows.")
                st.dataframe(load_csv(str(stability_path)), width="stretch", hide_index=True)
    st.subheader("Customer records")
    search_id = st.text_input("Find customer by Customer ID", placeholder="Enter an ID or leave blank")
    if search_id.strip():
        view = view[view["Customer ID"].astype(str).str.contains(search_id.strip(), regex=False)]
    st.dataframe(view.sort_values("Monetary", ascending=False), width="stretch", hide_index=True)
    st.caption("Segment numbers are cluster identifiers. Interpret them using the profile metrics above, not as an ordinal ranking.")

# ---------- Inactivity ----------
elif page == "Inactivity Prediction":
    st.title("Inactivity Prediction")
    st.write("Explore saved customer inactivity-risk scores. No predictions are recalculated in the dashboard.")
    scores = read_artifact("artifacts/inactivity/inactivity_v2_customer_scores.csv")
    meta = read_metadata("artifacts/inactivity/inactivity_model_v2_metadata.json")
    scores["Snapshot"] = pd.to_datetime(scores["Snapshot"], errors="coerce")
    snapshots = sorted(scores["Snapshot"].dropna().unique())
    if not snapshots:
        st.warning("No valid snapshot dates were found in the score artifact.")
        st.stop()
    selected_snapshot = st.selectbox("Snapshot date", snapshots, index=len(snapshots)-1, format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))
    score_variant = st.radio("Risk score view", ["In-window features", "12-month lookback features"], horizontal=True)
    if score_variant == "In-window features":
        score_col, band_col, pct_col = "InactivityRiskScore_inwindow", "RiskBand_inwindow", "RiskPercentile_inwindow"
        description = "Uses features computed within the six-month observation window."
    else:
        score_col, band_col, pct_col = "InactivityRiskScore_lookback12m", "RiskBand_lookback12m", "RiskPercentile_lookback12m"
        description = "Uses the extended 12-month lookback score variant."
    view = scores[scores["Snapshot"] == pd.Timestamp(selected_snapshot)].copy()
    view[score_col] = pd.to_numeric(view[score_col], errors="coerce")
    view = view.dropna(subset=[score_col])
    st.caption(description)
    st.caption("Scores rank customers well, but they are not exact probabilities: our evaluation found the average score can sit above or below the true inactivity rate as the season changes. Use the risk bands and percentiles to prioritise customers rather than reading the score as a literal chance.")
    counts = view[band_col].fillna("Unknown").value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", f"{view['Customer ID'].nunique():,}")
    c2.metric("Average risk score (uncalibrated)", f"{view[score_col].mean():.3f}")
    c3.metric("High-risk customers", f"{int((view[band_col] == 'High').sum()):,}")
    l, r = st.columns(2)
    with l:
        st.subheader("Risk-band distribution")
        st.bar_chart(counts.rename("Customers"))
    with r:
        st.subheader("Risk score distribution")
        hist = pd.cut(view[score_col], bins=10, include_lowest=True).value_counts().sort_index()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist.rename("Customers"))
    st.subheader("Customer risk lookup")
    query = st.text_input("Search Customer ID", placeholder="Enter customer ID")
    if query.strip():
        view = view[view["Customer ID"].astype(str).str.contains(query.strip(), regex=False)]
    st.dataframe(view.sort_values(score_col, ascending=False), width="stretch", hide_index=True)
    research_dir = ROOT / "artifacts" / "research_improvements"
    with st.expander("Research validation: baselines, calibration and missing-data subgroups"):
        for filename, caption in [
            ("inactivity_recency_baseline_comparison.csv", "Recency-only baseline comparison; raw recency is a ranking score."),
            ("inactivity_calibration_before_after.csv", "Calibration fit on April-May 2011 development predictions; evaluated on June-July 2011."),
            ("inactivity_missingness_subgroup_metrics.csv", "Development out-of-fold performance by prior-history and tenure subgroup."),
            ("inactivity_feature_missingness_audit.csv", "Missingness rates across the complete customer-snapshot table.")
        ]:
            fpath = research_dir / filename
            if fpath.exists():
                st.caption(caption)
                st.dataframe(load_csv(str(fpath)), width="stretch", hide_index=True)
    with st.expander("Model context and limitations"):
        st.write(meta.get("description", "Saved inactivity prediction model."))
        st.write("Target definition:", meta.get("target_definition", "Not specified in metadata."))
        st.write("Observation window (months):", meta.get("observation_months", "Not specified"))
        st.write("Prediction horizon (months):", meta.get("prediction_months", "Not specified"))
        st.write("These are historical snapshot scores from the supplied artifact. They should not be interpreted as current real-time customer risk.")

# ---------- Associations ----------
elif page == "Product Associations":
    st.title("Product Association Mining")
    st.write("Explore frequent co-purchase rules generated from invoice-level baskets.")
    rules = read_artifact("artifacts/association/association_rules.csv")
    use_filtered = st.checkbox("Exclude potential same-series rules (experimental description-based filter)", value=False)
    if use_filtered:
        filtered_path = ROOT / "artifacts" / "research_improvements" / "association_rules_conservative_filtered.csv"
        if filtered_path.exists():
            rules = load_csv(str(filtered_path))
            st.caption("Experimental filter: code proximity plus product-description evidence. Review the audit before treating a flagged rule as trivial.")
    meta = read_metadata("artifacts/association/association_run_summary.json")
    mapping = load_product_mapping(str(MAPPING_FILE))
    c1, c2, c3 = st.columns(3)
    c1.metric("Invoice baskets", f"{int(meta.get('invoice_baskets', 0)):,}")
    c2.metric("Unique products", f"{int(meta.get('unique_products', 0)):,}")
    c3.metric("Rules", f"{len(rules):,}")

    min_lift = st.slider("Minimum lift", 1.0, max(2.0, float(rules["lift"].max())), 1.0, 0.5)
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.2, 0.05)
    filtered = rules[(rules["lift"] >= min_lift) & (rules["confidence"] >= min_conf)].copy()
    filtered = filtered.sort_values(["confidence", "lift", "support"], ascending=False)
    st.caption(f"Showing {len(filtered):,} of {len(rules):,} rules matching the selected thresholds.")
    display = filtered.head(200).copy()
    display["Antecedent products"] = display["antecedent_codes"].apply(lambda s: ", ".join(pretty_product(x, mapping) for x in as_codes(s)))
    display["Consequent products"] = display["consequent_codes"].apply(lambda s: ", ".join(pretty_product(x, mapping) for x in as_codes(s)))
    cols = ["Antecedent products", "Consequent products", "support", "confidence", "lift"]
    st.dataframe(display[cols], width="stretch", hide_index=True)
    st.download_button("Download filtered rules", filtered.to_csv(index=False).encode("utf-8"), "filtered_association_rules.csv", "text/csv")
    st.caption("Product associations indicate co-occurrence, not causation or guaranteed demand.")

# ---------- Recommendations ----------
elif page == "Recommendations":
    st.title("Basket Recommendations")
    st.write("Enter products in a basket to retrieve recommendations from the saved association rules.")
    rules = read_artifact("artifacts/association/association_rules.csv")
    use_filtered = st.checkbox("Use conservative product-family filtered rules (experimental)", value=False)
    if use_filtered:
        filtered_path = ROOT / "artifacts" / "research_improvements" / "association_rules_conservative_filtered.csv"
        if filtered_path.exists():
            rules = load_csv(str(filtered_path))
            st.caption("This removes candidate same-series rules using a heuristic; compare against the unfiltered output.")
    mapping = load_product_mapping(str(MAPPING_FILE))
    rules = rules.copy()
    rules["antecedent_set"] = rules["antecedent_codes"].apply(as_codes)
    rules["consequent_set"] = rules["consequent_codes"].apply(as_codes)
    rules["confidence"] = pd.to_numeric(rules["confidence"], errors="coerce")
    rules["lift"] = pd.to_numeric(rules["lift"], errors="coerce")
    rules["support"] = pd.to_numeric(rules["support"], errors="coerce")

    codes = sorted(mapping.keys())
    selected_labels = st.multiselect(
        "Products currently in the basket",
        options=codes,
        format_func=lambda code: pretty_product(code, mapping),
        help="Search by product description or stock code."
    )
    top_k = st.slider("Number of recommendations", 1, 20, 5)
    if st.button("Generate recommendations", type="primary", disabled=not selected_labels):
        basket = set(selected_labels)
        candidates = []
        for _, rule in rules.iterrows():
            antecedent = set(rule["antecedent_set"])
            consequent = set(rule["consequent_set"])
            if antecedent and antecedent.issubset(basket):
                for code in consequent - basket:
                    candidates.append({
                        "Stock code": code,
                        "Product": pretty_product(code, mapping),
                        "Confidence": rule["confidence"],
                        "Lift": rule["lift"],
                        "Support": rule["support"],
                        "Matched products": ", ".join(pretty_product(x, mapping) for x in sorted(antecedent)),
                    })
        if not candidates:
            st.warning("No matching association rules were found for this basket. Try a different combination of products.")
        else:
            result = pd.DataFrame(candidates)
            result = (result.sort_values(["Confidence", "Lift", "Support"], ascending=False)
                            .drop_duplicates("Stock code", keep="first")
                            .head(top_k))
            st.success(f"Generated {len(result)} recommendations.")
            st.dataframe(result, width="stretch", hide_index=True)
            st.download_button("Download recommendations", result.to_csv(index=False).encode("utf-8"), "basket_recommendations.csv", "text/csv")
    st.divider()
    st.subheader("Offline evaluation")
    eval_df = read_artifact("artifacts/recommendation/recommendation_evaluation.csv")
    st.caption("Temporal holdout evaluation saved by Module 4. These metrics describe the offline test, not live conversion performance.")
    st.dataframe(eval_df, width="stretch", hide_index=True)

# ---------- About ----------
elif page == "About & Methodology":
    st.title("About RetailPulse")
    st.write("RetailPulse is a retail customer analytics project combining customer segmentation, inactivity-risk scoring, product association mining, and basket-based recommendations.")
    st.subheader("Project modules")
    modules = [
        ("Customer Segmentation", "Saved K-means customer assignments using Recency, Frequency, and Monetary features."),
        ("Inactivity Prediction", "Saved inactivity-risk scores from two models: a logistic regression on six-month in-window features and a histogram gradient boosting model that adds a 12-month look-back. The target is no purchase in the next three calendar months."),
        ("Product Association Mining", "Apriori and FP-Growth frequent itemsets and association rules based on invoice baskets."),
        ("Recommendation Engine", "Basket recommendations derived from association rules and ranked by confidence, lift, then support."),
        ("Dashboard", "Interactive exploration of the saved outputs and on-demand basket recommendations."),
    ]
    for name, detail in modules:
        with st.expander(name):
            st.write(detail)
    st.subheader("Research improvements and reproducibility")
    report_path = ROOT / "docs" / "research_improvements" / "EIGHT_IMPROVEMENTS_EXECUTION_REPORT.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    count_path = ROOT / "artifacts" / "research_improvements" / "dataset_count_reconciliation.csv"
    if count_path.exists():
        st.caption("Raw-source and cleaned module populations are reconciled below.")
        st.dataframe(load_csv(str(count_path)), width="stretch", hide_index=True)
    st.subheader("Important interpretation notes")
    st.markdown("""
    - Cluster IDs are labels, not ordered customer-value grades.
    - Inactivity risk scores are tied to historical snapshot dates and selected feature variants.
    - Association rules describe co-purchase patterns; they do not establish causality.
    - Precision@K, Recall@K, and Coverage@K are offline temporal-holdout metrics.
    - The dashboard reads existing artifacts; it does not retrain the project's models.
    """)
    st.caption("RetailPulse | Customer intelligence from historical online retail transactions")
