"""
RetailPulse research-improvement audit and evaluation utilities.

Run from the project root:
  python scripts/run_research_improvements.py

Optional, when the matching Online Retail II processed dataset is available:
  python scripts/run_research_improvements.py --transactions data/processed/cleaned_transactions.csv

Optional development-only prediction file (must contain y, p and preferably Origin):
  python scripts/run_research_improvements.py --development-predictions path/to/development_predictions.csv

IMPORTANT: never fit calibration on locked/previously inspected holdout predictions.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, brier_score_loss, log_loss,
                             precision_score, recall_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ART / "research_improvements"
OUT.mkdir(parents=True, exist_ok=True)

def write_csv(df, name):
    df.to_csv(OUT / name, index=False)
    return df

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def recommendation_denominators():
    p = ART / "recommendation" / "recommendation_evaluation.csv"
    if not p.exists(): return
    d = pd.read_csv(p)
    rows = []
    for r in d.to_dict("records"):
        n, covered, k = int(r["Eligible_test_invoices"]), int(r["Invoices_with_recommendations"]), int(r["K"])
        coverage = covered / n if n else np.nan
        all_precision = float(r["Precision@K"])
        conditional = all_precision / coverage if coverage else np.nan
        rows.append({"K": k, "Eligible_baskets": n, "Baskets_with_recommendations": covered,
                     "No_recommendation_baskets": n-covered, "Coverage": coverage,
                     "Precision_at_K_all_baskets": all_precision,
                     "Precision_at_K_covered_baskets": conditional,
                     "Recall_at_K_all_baskets": float(r["Recall@K"]),
                     "Interpretation": "All-basket precision counts no-suggestion baskets as zero; covered-only precision conditions on successful coverage."})
    write_csv(pd.DataFrame(rows), "recommendation_denominator_comparison.csv")

def association_diagnostics():
    p = ART / "association" / "association_rules.csv"
    if not p.exists(): return
    d = pd.read_csv(p, dtype={"antecedent_codes":str,"consequent_codes":str})
    def split(s): return [x.strip() for x in str(s).split("|") if x.strip()]
    # Product-code proximity is only a screening proxy, not proof of a product series.
    def numeric_codes(s):
        vals=[]
        for x in split(s):
            try: vals.append(int(float(x)))
            except Exception: pass
        return vals
    def proximity(row):
        vals=numeric_codes(row["antecedent_codes"])+numeric_codes(row["consequent_codes"])
        return len(vals)>=2 and all(len(str(v))==5 for v in vals) and (max(vals)-min(vals)<=5)
    d["antecedent_size"] = d["antecedent_codes"].map(lambda x:len(split(x)))
    d["consequent_size"] = d["consequent_codes"].map(lambda x:len(split(x)))
    d["numeric_code_proximity_flag"] = d.apply(proximity,axis=1)
    d["rule_family_screen"] = np.where(d["numeric_code_proximity_flag"],
        "Potential related-code family (manual description verification required)",
        "Not flagged by numeric proximity heuristic")
    d.sort_values(["numeric_code_proximity_flag","lift"],ascending=[False,False]).to_csv(OUT/"association_rule_diagnostics.csv",index=False)
    summary = pd.DataFrame([
      {"diagnostic":"Total saved rules","count":len(d),"share":1.0},
      {"diagnostic":"Potential related-code family (heuristic only)","count":int(d.numeric_code_proximity_flag.sum()),"share":float(d.numeric_code_proximity_flag.mean()) if len(d) else np.nan}
    ])
    write_csv(summary,"association_rule_diagnostic_summary.csv")

def inactivity_diagnostics(dev_path=None, test_path=None):
    # Existing locked predictions are retained for reporting, never used to fit a calibrator.
    p = ART/"inactivity"/"inactivity_v2_locked_predictions.csv"
    if p.exists():
        d=pd.read_csv(p)
        rows=[]
        for model,g in d.groupby("Model"):
            if g.y.nunique()>1:
                rows.append({"Model":model,"Scope":"Previously saved locked/confirmatory predictions",
                  "ROC_AUC":roc_auc_score(g.y,g.p),"Brier":brier_score_loss(g.y,g.p),
                  "Mean_predicted_probability":g.p.mean(),"Observed_inactivity_rate":g.y.mean(),
                  "Calibration_fit_performed":False,
                  "Note":"Descriptive only. Do not fit calibration on this locked/previously inspected data."})
        write_csv(pd.DataFrame(rows),"inactivity_locked_diagnostics.csv")
    if dev_path:
        dev=pd.read_csv(dev_path)
        required={"y","p"}
        if not required.issubset(dev.columns): raise ValueError("Development predictions must contain y and p.")
        # Calibrate on development predictions, and export cross-validated-style in-sample
        # diagnostic with explicit warning; user must evaluate calibrated mapping on later untouched data.
        y=dev.y.astype(int).to_numpy(); p0=np.clip(dev.p.to_numpy(float),1e-6,1-1e-6)
        iso=IsotonicRegression(out_of_bounds="clip").fit(p0,y)
        calibrated=iso.predict(p0)
        # Platt scaling on logit score
        x=np.log(p0/(1-p0)).reshape(-1,1)
        platt=LogisticRegression().fit(x,y)
        platt_p=platt.predict_proba(x)[:,1]
        res=pd.DataFrame([
          {"Method":"Uncalibrated","Brier":brier_score_loss(y,p0),"LogLoss":log_loss(y,p0),"MeanProbability":p0.mean(),"ObservedRate":y.mean(),"FitScope":"development only; apparent fit"},
          {"Method":"Isotonic","Brier":brier_score_loss(y,calibrated),"LogLoss":log_loss(y,calibrated),"MeanProbability":calibrated.mean(),"ObservedRate":y.mean(),"FitScope":"development only; apparent fit; evaluate on untouched later period"},
          {"Method":"Platt","Brier":brier_score_loss(y,platt_p),"LogLoss":log_loss(y,platt_p),"MeanProbability":platt_p.mean(),"ObservedRate":y.mean(),"FitScope":"development only; apparent fit; evaluate on untouched later period"}
        ])
        write_csv(res,"inactivity_calibration_development_fit.csv")
        import joblib
        joblib.dump({"isotonic":iso,"platt":platt,"feature":"raw_probability_logit",
                     "fit_scope":"development predictions only"},OUT/"inactivity_calibrators.joblib")
        if test_path and Path(test_path).exists():
            test=pd.read_csv(test_path)
            if not {"y","p"}.issubset(test.columns):
                raise ValueError("Calibration test predictions must contain y and p.")
            yt=test.y.astype(int).to_numpy()
            pt=np.clip(test.p.to_numpy(float),1e-6,1-1e-6)
            iso_test=iso.predict(pt)
            platt_test=platt.predict_proba(np.log(pt/(1-pt)).reshape(-1,1))[:,1]
            rows=[]
            for name,pp in [("Uncalibrated",pt),("Isotonic",iso_test),("Platt",platt_test)]:
                rows.append({"Method":name,"Test_rows":len(yt),"ROC_AUC":roc_auc_score(yt,pp),
                  "Brier":brier_score_loss(yt,pp),"LogLoss":log_loss(yt,pp),
                  "Mean_probability":pp.mean(),"Observed_rate":yt.mean(),
                  "Evaluation_scope":"Untouched later test; calibrator fitted on development only"})
            write_csv(pd.DataFrame(rows),"inactivity_calibration_untouched_test.csv")

def segmentation_diagnostics():
    p=ART/"segmentation"/"customer_segments.csv"
    if not p.exists(): return
    d=pd.read_csv(p)
    cols=["Recency","Frequency","Monetary"]
    X=np.log1p(d[cols].clip(lower=0))
    X=StandardScaler().fit_transform(X)
    rows=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,random_state=42,n_init=20).fit(X)
        labels=km.labels_
        rows.append({"K":k,"Inertia":km.inertia_,"Silhouette":silhouette_score(X,labels),
                     "Davies_Bouldin":davies_bouldin_score(X,labels),
                     "Calinski_Harabasz":calinski_harabasz_score(X,labels),
                     "Source":"Recomputed from saved current customer feature table"})
    write_csv(pd.DataFrame(rows),"segmentation_cluster_count_comparison.csv")
    # Random-seed stability on identical observations (not temporal stability).
    stability=[]
    for k in range(2,9):
        ref=KMeans(n_clusters=k,random_state=42,n_init=20).fit_predict(X)
        for seed in [7,21,84,123,2026]:
            other=KMeans(n_clusters=k,random_state=seed,n_init=20).fit_predict(X)
            stability.append({"K":k,"Seed":seed,"ARI_vs_seed42":adjusted_rand_score(ref,other),
                              "Scope":"Same-data initialization stability; not temporal stability"})
    write_csv(pd.DataFrame(stability),"segmentation_seed_stability.csv")
    seg=d.copy()
    seg["Segment"]=seg["Segment"].astype(str)
    prof=seg.groupby("Segment").agg(Customers=("Customer ID","nunique"),
      Mean_Recency=("Recency","mean"),Mean_Frequency=("Frequency","mean"),
      Mean_Monetary=("Monetary","mean")).reset_index()
    write_csv(prof,"segmentation_feature_profiles.csv")


def recommendation_baseline(transactions_path):
    if not transactions_path or not Path(transactions_path).exists(): return
    from collections import Counter
    tx=pd.read_csv(transactions_path,usecols=["Invoice","StockCode","InvoiceDate","Quantity","Price"],
                   dtype={"Invoice":"string","StockCode":"string"},parse_dates=["InvoiceDate"])
    tx=tx.dropna(subset=["Invoice","StockCode","InvoiceDate"])
    tx=tx[(tx.Quantity>0)&(tx.Price>0)].copy()
    tx["StockCode"]=tx.StockCode.str.strip()
    dates=tx.groupby("Invoice").InvoiceDate.min().sort_values()
    cut=max(1,int(len(dates)*0.80))
    train_ids=set(dates.index[:cut]); test_ids=set(dates.index[cut:])
    train=tx[tx.Invoice.isin(train_ids)]
    test=tx[tx.Invoice.isin(test_ids)]
    known=set(train.StockCode.unique())
    popularity=Counter(train.StockCode)
    top_products=[p for p,_ in popularity.most_common()]
    rule_path=ART/"recommendation"/"evaluation_association_rules.csv"
    rules=pd.read_csv(rule_path,dtype=str) if rule_path.exists() else pd.DataFrame()
    parsed=[]
    if not rules.empty and {"antecedents","consequents"}.issubset(rules.columns):
        for _,r in rules.iterrows():
            a=set(str(r.antecedents).split(" | ")); c=set(str(r.consequents).split(" | "))
            parsed.append((a,c,float(r.get("confidence",0) or 0),float(r.get("lift",0) or 0),float(r.get("support",0) or 0)))
    rng=np.random.default_rng(42)
    cases=[]
    baskets=test.groupby("Invoice").StockCode.apply(lambda x:sorted(set(x)&known))
    for inv,items in baskets.items():
        if len(items)<2: continue
        items=list(items); rng.shuffle(items); n=max(1,len(items)//2)
        context=set(items[:n]); held=set(items[n:])
        if context and held: cases.append((context,held))
    rows=[]
    for method in ["Training-period top sellers","Training-only association rules"]:
        pvals=[]; covered_pvals=[]; rec=[]; covered=0
        for context,held in cases:
            if method=="Training-period top sellers":
                recs=[x for x in top_products if x not in context][:5]
            else:
                cand=[]
                for a,c,conf,lift,sup in parsed:
                    if a and a.issubset(context):
                        for product in c-context: cand.append((product,conf,lift,sup))
                cand.sort(key=lambda x:(-x[1],-x[2],-x[3]))
                recs=[]
                for item,*_ in cand:
                    if item not in recs: recs.append(item)
                    if len(recs)==5: break
            hits=len(set(recs)&held); pvals.append(hits/5); rec.append(hits/len(held))
            if recs: covered_pvals.append(hits/5)
            covered += bool(recs)
        n=len(cases)
        rows.append({"Method":method,"K":5,"Eligible_test_baskets":n,
          "Baskets_with_recommendations":covered,"Coverage":covered/n if n else np.nan,
          "Precision_at_5_all_baskets":float(np.mean(pvals)) if n else np.nan,
          "Precision_at_5_covered_baskets":float(np.mean(covered_pvals)) if covered_pvals else np.nan,
          "Recall_at_5":float(np.mean(rec)) if n else np.nan,
          "Temporal_split":"first 80% invoice dates train; final 20% test",
          "Caveat":"Recomputed from the supplied transactions. Compare only if this is the exact source/version used for the saved project artifacts."})
    write_csv(pd.DataFrame(rows),"recommendation_baseline_comparison.csv")


def inactivity_subgroup_evaluation(feature_path):
    if not feature_path or not Path(feature_path).exists(): return
    d=pd.read_csv(feature_path)
    if not {"y","p"}.issubset(d.columns): return
    rows=[]
    groups={}
    for c in ["PriorWindowInvoices","PriorWindowRecencyDays","ObservedTenureDays"]:
        if c in d:
            if c=="ObservedTenureDays":
                groups["Tenure: <= median"]=d[c] <= d[c].median()
                groups["Tenure: > median"]=d[c] > d[c].median()
            else:
                groups[f"{c}: missing"]=d[c].isna()
                groups[f"{c}: observed"]=d[c].notna()
    for label,mask in groups.items():
        g=d[mask]
        if len(g)>5 and g.y.nunique()>1:
            rows.append({"Subgroup":label,"Rows":len(g),"Prevalence":g.y.mean(),
                         "ROC_AUC":roc_auc_score(g.y,g.p),"Brier":brier_score_loss(g.y,g.p),
                         "Mean_probability":g.p.mean()})
    write_csv(pd.DataFrame(rows),"inactivity_missingness_subgroup_metrics.csv")

def temporal_segmentation_stability(transactions_path):
    if not transactions_path or not Path(transactions_path).exists(): return
    d=pd.read_csv(transactions_path, usecols=["Customer ID","Invoice","InvoiceDate","TransactionValue"],
                  parse_dates=["InvoiceDate"])
    d=d.dropna(subset=["Customer ID","InvoiceDate"])
    d=d[d["Customer ID"].notna()].copy()
    d["Customer ID"]=d["Customer ID"].astype(str)
    lo,hi=d.InvoiceDate.min(),d.InvoiceDate.max()
    mid=lo+(hi-lo)/2
    def rfm(x,end):
        z=x.groupby("Customer ID").agg(Last=("InvoiceDate","max"),Frequency=("Invoice","nunique"),Monetary=("TransactionValue","sum"))
        z["Recency"]=(end.normalize()-z.Last.dt.normalize()).dt.days
        return z[["Recency","Frequency","Monetary"]]
    a=rfm(d[d.InvoiceDate<=mid],mid)
    b=rfm(d[d.InvoiceDate>mid],hi)
    common=a.index.intersection(b.index)
    if len(common)<30: return
    def cluster(z):
        X=StandardScaler().fit_transform(np.log1p(z.clip(lower=0)))
        return KMeans(n_clusters=3,random_state=42,n_init=20).fit_predict(X)
    la=cluster(a.loc[common]); lb=cluster(b.loc[common])
    write_csv(pd.DataFrame([{"Window_1_end":str(mid.date()),"Window_2_end":str(hi.date()),
      "Shared_customers":len(common),"Adjusted_Rand_Index":adjusted_rand_score(la,lb),
      "Method":"Independent KMeans K=3 on first and second chronological halves",
      "Caveat":"Descriptive temporal stability; feature definitions and customer overlap must be checked for publication."}]),
      "segmentation_temporal_stability.csv")

def inactivity_recency_baseline(feature_path):
    if not feature_path or not Path(feature_path).exists(): return
    d=pd.read_csv(feature_path)
    needed={"Origin","Customer ID","y","Recency"}
    if not needed.issubset(d.columns):
        raise ValueError(f"Inactivity baseline file must contain {sorted(needed)}")
    rows=[]
    for origin,g in d.groupby("Origin"):
        if g.y.nunique()<2: continue
        # Smaller Recency means lower risk, so rank by recency directly.
        rows.append({"Origin":origin,"Baseline":"Recency-only ranking","ROC_AUC":roc_auc_score(g.y,g.Recency),
                     "Rows":len(g),"Observed_inactivity_rate":g.y.mean()})
    write_csv(pd.DataFrame(rows),"inactivity_recency_baseline.csv")

def dataset_reconciliation(transactions_path=None):
    rows=[]
    if transactions_path and Path(transactions_path).exists():
        d=pd.read_csv(transactions_path,low_memory=False)
        rows.append({"Stage":"Provided processed transaction file","Rows":len(d),
          "Invoices":d["Invoice"].nunique() if "Invoice" in d else np.nan,
          "Customers":d["Customer ID"].nunique() if "Customer ID" in d else np.nan,
          "Products":d["StockCode"].nunique() if "StockCode" in d else np.nan,
          "Notes":"Counts read from explicitly supplied processed file."})
    # Counts from existing artifacts are included as distinct populations, never conflated.
    meta=load_json(ART/"association"/"association_run_summary.json")
    if meta:
        rows.append({"Stage":"Association mining input (saved metadata)","Rows":meta.get("valid_purchase_lines"),
          "Invoices":meta.get("invoice_baskets"),"Customers":np.nan,"Products":meta.get("unique_products"),
          "Notes":"Saved association-module population; reconcile against source file before publication."})
    seg=ART/"segmentation"/"customer_segments.csv"
    if seg.exists():
        s=pd.read_csv(seg)
        rows.append({"Stage":"Segmentation output (saved artifact)","Rows":len(s),
          "Invoices":np.nan,"Customers":s["Customer ID"].nunique(),"Products":np.nan,
          "Notes":"Customer-level population after module-specific eligibility filters."})
    rec=load_json(ART/"recommendation"/"evaluation_metadata.json")
    if rec:
        rows.append({"Stage":"Recommendation temporal test cases","Rows":rec.get("eligible_test_invoices"),
          "Invoices":rec.get("eligible_test_invoices"),"Customers":np.nan,"Products":np.nan,
          "Notes":"Eligible recommendation evaluation baskets, not full raw-data rows."})
    write_csv(pd.DataFrame(rows),"dataset_count_reconciliation.csv")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--transactions",default=None,help="Canonical cleaned_transactions.csv from the same Online Retail II source used for existing artifacts.")
    ap.add_argument("--inactivity-feature-evaluation",default=None,help="Temporal evaluation CSV with Origin, Customer ID, y, Recency for a simple recency baseline.")
    ap.add_argument("--calibration-test-predictions",default=None,help="Later untouched test predictions with y and p; used only to evaluate development-fitted calibrators.")
    ap.add_argument("--development-predictions",default=None,help="Development-only inactivity predictions CSV with y and p; never pass locked holdout predictions.")
    args=ap.parse_args()
    recommendation_denominators()
    recommendation_baseline(args.transactions)
    inactivity_recency_baseline(args.inactivity_feature_evaluation)
    inactivity_subgroup_evaluation(args.inactivity_feature_evaluation)
    temporal_segmentation_stability(args.transactions)
    association_diagnostics()
    inactivity_diagnostics(args.development_predictions,args.calibration_test_predictions)
    segmentation_diagnostics()
    dataset_reconciliation(args.transactions)
    print("Research-improvement artifacts written to:",OUT)
    print("NOTE: baseline, subgroup, and temporal-transfer tests require matching source transactions and development feature/prediction tables. See docs/research_improvements/IMPLEMENTATION_NOTES.md.")

if __name__=="__main__": main()
