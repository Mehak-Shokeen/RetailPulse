from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data" / "processed"

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
        st.dataframe(profiles, use_container_width=True, hide_index=True)

    st.subheader("Recommendation evaluation")
    st.caption("Offline temporal holdout results from the saved evaluation; these are not live business outcomes.")
    st.dataframe(rec_eval, use_container_width=True, hide_index=True)

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
    st.dataframe(profiles, use_container_width=True, hide_index=True)
    st.subheader("Customer records")
    search_id = st.text_input("Find customer by Customer ID", placeholder="Enter an ID or leave blank")
    if search_id.strip():
        view = view[view["Customer ID"].astype(str).str.contains(search_id.strip(), regex=False)]
    st.dataframe(view.sort_values("Monetary", ascending=False), use_container_width=True, hide_index=True)
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
    counts = view[band_col].fillna("Unknown").value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", f"{view['Customer ID'].nunique():,}")
    c2.metric("Average risk score", f"{view[score_col].mean():.3f}")
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
    st.dataframe(view.sort_values(score_col, ascending=False), use_container_width=True, hide_index=True)
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
    meta = read_metadata("artifacts/association/association_run_summary.json")
    mapping = load_product_mapping(str(DATA / "cleaned_transactions.csv"))
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
    st.dataframe(display[cols], use_container_width=True, hide_index=True)
    st.download_button("Download filtered rules", filtered.to_csv(index=False).encode("utf-8"), "filtered_association_rules.csv", "text/csv")
    st.caption("Product associations indicate co-occurrence, not causation or guaranteed demand.")

# ---------- Recommendations ----------
elif page == "Recommendations":
    st.title("Basket Recommendations")
    st.write("Enter products in a basket to retrieve recommendations from the saved association rules.")
    rules = read_artifact("artifacts/association/association_rules.csv")
    mapping = load_product_mapping(str(DATA / "cleaned_transactions.csv"))
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
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button("Download recommendations", result.to_csv(index=False).encode("utf-8"), "basket_recommendations.csv", "text/csv")
    st.divider()
    st.subheader("Offline evaluation")
    eval_df = read_artifact("artifacts/recommendation/recommendation_evaluation.csv")
    st.caption("Temporal holdout evaluation saved by Module 4. These metrics describe the offline test, not live conversion performance.")
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

# ---------- Research improvements ----------
elif page == "Research Improvements":
    st.title("Research Improvements & Evaluation Audit")
    st.write("Auditable diagnostics for baselines, calibration, recommendation coverage, association-rule redundancy, segmentation, missingness, and dataset populations.")
    improvement_dir = ARTIFACTS / "research_improvements"
    st.info("Some experiments require the matching original Online Retail II transaction source and development-only prediction tables. See docs/research_improvements/IMPLEMENTATION_NOTES.md before interpreting missing outputs.")
    display_files = [
        ("Recommendation denominator comparison", "recommendation_denominator_comparison.csv"),
        ("Association rule diagnostics", "association_rule_diagnostic_summary.csv"),
        ("Inactivity locked diagnostics", "inactivity_locked_diagnostics.csv"),
        ("Segmentation cluster-count comparison", "segmentation_cluster_count_comparison.csv"),
        ("Segmentation initialization stability", "segmentation_seed_stability.csv"),
        ("Segmentation feature profiles", "segmentation_feature_profiles.csv"),
        ("Dataset count reconciliation", "dataset_count_reconciliation.csv"),
        ("Source compatibility audit", "source_compatibility_audit.csv"),
        ("Inactivity recency baseline", "inactivity_recency_baseline.csv"),
        ("Inactivity missingness subgroups", "inactivity_missingness_subgroup_metrics.csv"),
        ("Recommendation baseline comparison", "recommendation_baseline_comparison.csv"),
        ("Temporal segmentation stability", "segmentation_temporal_stability.csv"),
    ]
    for label, filename in display_files:
        path = improvement_dir / filename
        if path.exists():
            with st.expander(label, expanded=filename in ["recommendation_denominator_comparison.csv","dataset_count_reconciliation.csv"]):
                frame = load_csv(str(path))
                st.dataframe(frame, use_container_width=True, hide_index=True)
                st.download_button(f"Download {label.lower()}", frame.to_csv(index=False).encode("utf-8"), filename, "text/csv", key=f"download_{filename}")
    st.caption("Calibration is fit only on development predictions. Locked or previously inspected holdout data must not be used to fit calibration.")

# ---------- About ----------
elif page == "About & Methodology":
    st.title("About RetailPulse")
    st.write("RetailPulse is a retail customer analytics project combining customer segmentation, inactivity-risk scoring, product association mining, and basket-based recommendations.")
    st.subheader("Project modules")
    modules = [
        ("Customer Segmentation", "Saved K-means customer assignments using Recency, Frequency, and Monetary features."),
        ("Inactivity Prediction", "Saved logistic-regression inactivity-risk scores. The target is no purchase in the next three calendar months."),
        ("Product Association Mining", "Apriori and FP-Growth frequent itemsets and association rules based on invoice baskets."),
        ("Recommendation Engine", "Basket recommendations derived from association rules and ranked by confidence, lift, then support."),
        ("Dashboard", "Interactive exploration of the saved outputs and on-demand basket recommendations."),
    ]
    for name, detail in modules:
        with st.expander(name):
            st.write(detail)
    st.subheader("Important interpretation notes")
    st.markdown("""
    - Cluster IDs are labels, not ordered customer-value grades.
    - Inactivity risk scores are tied to historical snapshot dates and selected feature variants.
    - Association rules describe co-purchase patterns; they do not establish causality.
    - Precision@K, Recall@K, and Coverage@K are offline temporal-holdout metrics.
    - The dashboard reads existing artifacts; it does not retrain the project's models.
    """)
    st.caption("RetailPulse | Customer intelligence from historical online retail transactions")
