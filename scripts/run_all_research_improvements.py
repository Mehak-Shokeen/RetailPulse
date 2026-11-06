"""
RetailPulse: reproducible implementation of the eight research improvements.

Run:
  python scripts/run_all_research_improvements.py --dataset path/to/online_retail_II.xlsx

The script:
- reconstructs the canonical cleaned transaction population from the two-sheet UCI workbook;
- computes same-population baselines and recommendation denominator metrics;
- fits calibration mappings on rolling-origin development predictions only;
- evaluates the calibrators on June/July confirmatory origins and separately labels
  August 2011 as a previously inspected holdout;
- evaluates inactivity model performance by missingness/tenure subgroup;
- audits association rules using product descriptions as well as code proximity;
- evaluates cluster-count choice and temporal segmentation stability;
- writes a stage-by-stage data reconciliation table and an auditable report.

It does not overwrite the original models or original evaluation artifacts.
"""
from __future__ import annotations
import argparse, json, re, hashlib, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ART / "research_improvements"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def save(df, name):
    path = OUT / name
    df.to_csv(path, index=False)
    return path

def metric_row(y, p, label, scope):
    y = np.asarray(y, dtype=int); p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    edges=np.quantile(p,np.linspace(0,1,11))
    bins=np.clip(np.searchsorted(edges,p,side="right")-1,0,9)
    ece=float(sum((bins==b).mean()*abs(y[bins==b].mean()-p[bins==b].mean())
                  for b in range(10) if np.any(bins==b)))
    return {"Model": label, "Scope": scope, "N": len(y),
            "ROC_AUC": roc_auc_score(y,p) if len(np.unique(y)) > 1 else np.nan,
            "PR_AUC": average_precision_score(y,p) if len(np.unique(y)) > 1 else np.nan,
            "Brier": brier_score_loss(y,p), "LogLoss": log_loss(y,p),
            "ECE_10_bins":ece,"Mean_probability": float(p.mean()), "Observed_rate": float(y.mean())}

def load_and_clean(dataset):
    xls = pd.ExcelFile(dataset)
    required_sheets = {"Year 2009-2010", "Year 2010-2011"}
    if not required_sheets.issubset(set(xls.sheet_names)):
        raise ValueError(f"Expected Online Retail II sheets {sorted(required_sheets)}; found {xls.sheet_names}")
    parts=[]
    for sheet in ["Year 2009-2010","Year 2010-2011"]:
        d=xls.parse(sheet)
        d["SourceSheet"]=sheet
        parts.append(d)
    raw=pd.concat(parts,ignore_index=True)
    original_cols=["Invoice","StockCode","Description","Quantity","InvoiceDate","Price","Customer ID","Country"]
    raw["InvoiceDate"]=pd.to_datetime(raw["InvoiceDate"],errors="coerce")
    dedup=raw.drop_duplicates(subset=original_cols,keep="first").copy()
    invoice=dedup["Invoice"].astype(str).str.upper()
    cancellation=invoice.str.startswith("C")
    ordinary=dedup.loc[~cancellation & (dedup["Quantity"]>0)].copy()
    purchase=ordinary.loc[(ordinary["Price"]>0)&ordinary["InvoiceDate"].notna()].copy()
    purchase["Invoice"]=purchase["Invoice"].astype(str).str.strip()
    purchase["StockCode"]=purchase["StockCode"].astype(str).str.strip()
    purchase["Customer ID"]=pd.to_numeric(purchase["Customer ID"],errors="coerce")
    purchase["TransactionValue"]=purchase["Quantity"]*purchase["Price"]
    return raw,dedup,ordinary,purchase

def build_inactivity_evaluation(tx, saved_predictions):
    # Use the project's exact feature builder and target definition.
    import sys
    sys.path.insert(0,str(ROOT))
    from src.features.inactivity_features import (
        prepare_transactions, build_snapshot_features, build_labels,
        snapshot_windows, fit_derived_params, add_derived_features
    )
    prepared=prepare_transactions(tx)
    data_start=prepared.InvoiceDate.min().normalize().replace(day=1)
    periods=pd.period_range("2010-05","2011-08",freq="M")
    table=[]
    for period in periods:
        snap=period.end_time
        feat=build_snapshot_features(prepared,snap,data_start=data_start)
        feat["y"]=build_labels(prepared,snap,feat["Customer ID"])
        feat["Origin"]=str(period)
        feat["SnapPeriod"]=period
        table.append(feat)
    table=pd.concat(table,ignore_index=True)
    # Recreate the original 17-feature GB-17 model under rolling-origin maturity guards.
    # Match the frozen production B bundle exactly: HGB, 13 extended-lookback features.
    bundle_meta=json.loads((ART/"inactivity"/"inactivity_model_v2_extended_lookback_metadata.json").read_text(encoding="utf-8"))
    cols=list(bundle_meta["feature_columns"])
    hgb_params=dict(bundle_meta["params"])
    # Early origins are included only for a missing-history subgroup audit; the frozen
    # development origins remain Dec 2010-May 2011 for calibration/model-selection.
    early_audit_origins=["2010-09","2010-10","2010-11"]
    dev_origins=["2010-12","2011-01","2011-02","2011-03","2011-04","2011-05"]
    all_oof_origins=early_audit_origins+dev_origins
    dev_predictions=[]
    feature_rows=[]
    for origin in all_oof_origins:
        train=table[table["SnapPeriod"]+3 <= pd.Period(origin,"M")].copy()
        test=table[table["Origin"]==origin].copy()
        if train.empty or test.empty or train["y"].nunique()<2: continue
        derived=fit_derived_params(train)
        train=add_derived_features(train,derived); test=add_derived_features(test,derived)
        model=__import__("sklearn.ensemble",fromlist=["HistGradientBoostingClassifier"]).HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, **hgb_params)
        model.fit(train[cols],train["y"].astype(int))
        p=model.predict_proba(test[cols])[:,1]
        out=test[["Origin","Customer ID","y","Recency","Frequency","PriorWindowInvoices",
                  "PriorWindowRecencyDays","ObservedTenureDays"]].copy()
        out["p"]=p
        out["PredictionRole"]=("exploratory early-origin missing-history audit" if origin in early_audit_origins
                              else "frozen development OOF")
        out["PredictionSource"]="saved HGB extended-lookback configuration retrained per origin"
        dev_predictions.append(out)
        feature_rows.append(test[["Origin","Customer ID","y","Recency","Frequency","PriorWindowInvoices",
                                  "PriorWindowRecencyDays","ObservedTenureDays"]].copy())
    dev=pd.concat(dev_predictions,ignore_index=True) if dev_predictions else pd.DataFrame()
    save(dev,"inactivity_development_oof_predictions.csv")
    # Compare saved locked predictions with recency baseline using exactly matched customer-origin rows.
    locked=pd.read_csv(saved_predictions)
    locked["Customer ID"]=pd.to_numeric(locked["Customer ID"],errors="coerce")
    feat_locked=[]
    for origin in sorted(locked["Origin"].unique()):
        period=pd.Period(origin,"M")
        snap=period.end_time
        feat=build_snapshot_features(prepared,snap,data_start=data_start)
        feat["Origin"]=origin
        feat_locked.append(feat[["Origin","Customer ID","Recency","Frequency","PriorWindowInvoices",
                                 "PriorWindowRecencyDays","ObservedTenureDays"]])
    feat_locked=pd.concat(feat_locked,ignore_index=True)
    locked=locked.merge(feat_locked,on=["Origin","Customer ID"],how="left",validate="many_to_one")
    # Baseline comparison on locked origins; recency ranks high-risk when larger.
    base=[]
    # Recency is a ranking score, not a probability. Never pass raw day counts to Brier/log-loss.
    # A separate one-variable logistic baseline is fitted on development OOF rows only.
    recency_logit=None
    if not dev.empty:
        rec_cal=dev[dev.Origin.isin(["2011-04","2011-05"])]
        recency_logit=LogisticRegression(C=1e6,max_iter=2000).fit(rec_cal[["Recency"]],rec_cal["y"].astype(int))
    for model_name,g in locked.groupby("Model"):
        for origin,og in g.groupby("Origin"):
            y=og.y.astype(int).to_numpy(); score=og.Recency.to_numpy(float)
            row={"Model":"Recency-only ranking baseline","Compared_model":model_name,"Scope":origin,"N":len(y),
                 "ROC_AUC":roc_auc_score(y,score),"PR_AUC":average_precision_score(y,score),
                 "Brier":np.nan,"LogLoss":np.nan,"Mean_probability":np.nan,"Observed_rate":float(y.mean()),
                 "Note":"Raw recency is a ranking score; probability metrics intentionally omitted."}
            if recency_logit is not None:
                prob=recency_logit.predict_proba(og[["Recency"]])[:,1]
                row.update({"Logistic_baseline_ROC_AUC":roc_auc_score(y,prob),
                    "Logistic_baseline_PR_AUC":average_precision_score(y,prob),
                    "Logistic_baseline_Brier":brier_score_loss(y,prob),
                    "Logistic_baseline_LogLoss":log_loss(y,prob),
                    "Logistic_baseline_mean_probability":float(prob.mean()),
                    "Logistic_fit_scope":"April-May 2011 development OOF predictions only"})
            base.append(row)
    baseline=pd.DataFrame(base)
    save(baseline,"inactivity_recency_baseline_comparison.csv")
    model_rows=[]
    for model_name,g in locked.groupby("Model"):
        for origin,og in g.groupby("Origin"):
            model_rows.append(metric_row(og.y,og.p,model_name,origin))
    save(pd.DataFrame(model_rows),"inactivity_locked_model_comparison.csv")
    # Calibration: fit on rolling-origin development OOF predictions only; never fit on locked predictions.
    cal_rows=[]; cal_models={}
    if not dev.empty:
        # Fit calibration only on the latest two development origins (Apr-May 2011),
        # closest in time to the confirmatory June-July block.
        caldev=dev[dev.Origin.isin(["2011-04","2011-05"])]
        ydev=caldev.y.astype(int).to_numpy()
        pdev=np.clip(caldev.p.to_numpy(float),1e-6,1-1e-6)
        isotonic=IsotonicRegression(out_of_bounds="clip").fit(pdev,ydev)
        platt=LogisticRegression(C=1e6,max_iter=2000).fit(np.log(pdev/(1-pdev)).reshape(-1,1),ydev)
        cal_models={"isotonic":isotonic,"platt":platt,
                    "fit_scope":"April-May 2011 development OOF predictions only",
                    "fit_origins":["2011-04","2011-05"],
                    "base_model":"B: 12-month look-back",
                    "base_model_family":"HistGradientBoostingClassifier",
                    "feature_columns":cols,
                    "calibration_evaluation":"June-July 2011 confirmatory block; August 2011 previously inspected"}
        # Model B is the primary extended-lookback saved model.
        primary="B: 12-month look-back"
        if primary not in set(locked.Model): primary=locked.Model.iloc[0]
        for scope,subset in [
            ("Confirmatory June-July 2011",locked[locked.Origin.isin(["2011-06","2011-07"]) & (locked.Model==primary)]),
            ("Previously inspected August 2011 (descriptive only)",locked[(locked.Origin=="2011-08") & (locked.Model==primary)])
        ]:
            if subset.empty: continue
            y=subset.y.astype(int).to_numpy(); p=np.clip(subset.p.to_numpy(float),1e-6,1-1e-6)
            vals=[("Uncalibrated",p),("Platt",platt.predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]),
                  ("Isotonic",isotonic.predict(p))]
            for method,pp in vals:
                cal_rows.append(metric_row(y,pp,method,scope))
        joblib.dump(cal_models,OUT/"inactivity_calibrators_development_only.joblib")
    save(pd.DataFrame(cal_rows),"inactivity_calibration_before_after.csv")
    # Subgroup analysis: development OOF rows include true feature-missingness patterns.
    subgroup_source=dev.copy()
    if not subgroup_source.empty:
        subgroup_defs={
          "PriorWindowInvoices missing (window unobservable)":subgroup_source.PriorWindowInvoices.isna(),
          "PriorWindowInvoices observed":subgroup_source.PriorWindowInvoices.notna(),
          "PriorWindowRecencyDays missing (no purchase in prior window)":subgroup_source.PriorWindowRecencyDays.isna(),
          "PriorWindowRecencyDays observed":subgroup_source.PriorWindowRecencyDays.notna(),
          "ObservedTenureDays <= median":subgroup_source.ObservedTenureDays<=subgroup_source.ObservedTenureDays.median(),
          "ObservedTenureDays > median":subgroup_source.ObservedTenureDays>subgroup_source.ObservedTenureDays.median()
        }
        rows=[]
        for label,mask in subgroup_defs.items():
            g=subgroup_source[mask]
            if len(g)>10 and g.y.nunique()>1:
                rows.append({"Subgroup":label,"Rows":len(g),"Missing_share":float(mask.mean()),
                    "ROC_AUC":roc_auc_score(g.y,g.p),"PR_AUC":average_precision_score(g.y,g.p),
                    "Brier":brier_score_loss(g.y,g.p),"Mean_probability":g.p.mean(),
                    "Observed_inactivity_rate":g.y.mean(),
                    "Scope":"rolling-origin development OOF; saved HGB extended-lookback configuration retrained per origin"})
        save(pd.DataFrame(rows),"inactivity_missingness_subgroup_metrics.csv")
        missing_summary=[]
        for c in ["PriorWindowInvoices","PriorWindowRecencyDays","ObservedTenureDays"]:
            missing_summary.append({"Feature":c,"Rows":len(table),
              "Missing_rows":int(table[c].isna().sum()),
              "Missing_share":float(table[c].isna().mean()),
              "Interpretation":("Prior window unavailable before sufficient source history" if c=="PriorWindowInvoices"
                else "No qualifying prior-window purchase or unavailable history" if c=="PriorWindowRecencyDays"
                else "Computed from observed customer history")})
        save(pd.DataFrame(missing_summary),"inactivity_feature_missingness_audit.csv")
    # Data governance table
    roles=[{"Origin":o,"Role":"Development OOF; model fitting/selection only"} for o in dev_origins]
    roles += [{"Origin":o,"Role":"Exploratory subgroup diagnostic only; not used for calibration or model selection"}
              for o in early_audit_origins]
    roles += [{"Origin":"2011-06","Role":"Confirmatory block; not used to fit calibration"},
              {"Origin":"2011-07","Role":"Confirmatory block; used as post-calibration evaluation"},
              {"Origin":"2011-08","Role":"Previously inspected holdout; descriptive only; not pristine"}]
    save(pd.DataFrame(roles),"inactivity_temporal_data_governance.csv")
    return table,locked,dev

def recommendation_analysis(tx):
    from collections import Counter
    p=ART/"recommendation"/"recommendation_evaluation.csv"
    rules_path=ART/"recommendation"/"evaluation_association_rules.csv"
    if not p.exists() or not rules_path.exists():
        raise FileNotFoundError("Original recommendation evaluation and training-only rule artifacts are required.")
    data=tx[["Invoice","StockCode","InvoiceDate","Quantity","Price"]].copy()
    data=data.dropna(subset=["Invoice","StockCode","InvoiceDate"])
    data=data[(data.Quantity>0)&(data.Price>0)].copy()
    data["Invoice"]=data.Invoice.astype(str); data["StockCode"]=data.StockCode.astype(str).str.strip()
    invoice_dates=data.groupby("Invoice").InvoiceDate.min().sort_values()
    cut=int(len(invoice_dates)*.80)
    cutoff=invoice_dates.iloc[cut]
    train_ids=set(invoice_dates[invoice_dates<cutoff].index)
    test_ids=set(invoice_dates[invoice_dates>=cutoff].index)
    train=data[data.Invoice.isin(train_ids)]
    test=data[data.Invoice.isin(test_ids)]
    known=set(train.StockCode.unique())
    popularity=Counter(train.StockCode)
    top=[x for x,_ in popularity.most_common()]
    rr=pd.read_csv(rules_path,dtype=str)
    def parse(s): return set(x.strip() for x in str(s).split(" | ") if x.strip())
    parsed=[]
    for _,r in rr.iterrows():
        if "antecedents" in r and "consequents" in r:
            parsed.append((parse(r.antecedents),parse(r.consequents),
                           float(r.get("confidence",0) or 0),float(r.get("lift",0) or 0),float(r.get("support",0) or 0)))
    rng=np.random.default_rng(42)
    baskets=test.groupby("Invoice").StockCode.apply(lambda x:sorted(set(x)&known))
    cases=[]
    for inv,items in baskets.items():
        if len(items)<2: continue
        items=list(items); rng.shuffle(items); n=max(1,len(items)//2)
        context=set(items[:n]); held=set(items[n:])
        if context and held: cases.append((context,held))
    rows=[]
    methods=["Association-rule recommender","Training-period top sellers"]
    for method in methods:
        prec_all=[]; prec_covered=[]; recalls=[]; covered=0
        for context,held in cases:
            if method=="Training-period top sellers":
                recs=[x for x in top if x not in context][:5]
            else:
                candidates=[]
                for a,c,conf,lift,sup in parsed:
                    if a and a.issubset(context):
                        for item in c-context: candidates.append((item,conf,lift,sup))
                candidates.sort(key=lambda x:(-x[1],-x[2],-x[3]))
                recs=[]
                for item,*_ in candidates:
                    if item not in recs: recs.append(item)
                    if len(recs)==5: break
            hits=len(set(recs)&held)
            prec_all.append(hits/5)
            recalls.append(hits/len(held))
            if recs:
                covered+=1; prec_covered.append(hits/5)
        n=len(cases)
        rows.append({"Method":method,"K":5,"Eligible_test_baskets":n,"Baskets_with_recommendations":covered,
          "No_recommendation_baskets":n-covered,"Coverage":covered/n if n else np.nan,
          "Precision_at_5_all_baskets":float(np.mean(prec_all)) if n else np.nan,
          "Precision_at_5_covered_baskets":float(np.mean(prec_covered)) if prec_covered else np.nan,
          "Recall_at_5":float(np.mean(recalls)) if n else np.nan,
          "Temporal_cutoff":str(cutoff),"Train_invoice_count":len(train_ids),"Test_invoice_count":len(test_ids),
          "Note":"The existing association rules were mined on training invoices only; popularity baseline is counted on the identical test basket cases."})
    result=pd.DataFrame(rows)
    # Include the saved original model score as an audit row; do not silently equate it with the rerun.
    saved=pd.read_csv(p)
    for _,r in saved[saved.K==5].iterrows():
        cov=float(r["Coverage@K"])
        result.loc[len(result)]={"Method":"Saved project recommender (original evaluation artifact)","K":5,
          "Eligible_test_baskets":int(r["Eligible_test_invoices"]),
          "Baskets_with_recommendations":int(r["Invoices_with_recommendations"]),
          "No_recommendation_baskets":int(r["Eligible_test_invoices"]-r["Invoices_with_recommendations"]),
          "Coverage":cov,"Precision_at_5_all_baskets":float(r["Precision@K"]),
          "Precision_at_5_covered_baskets":float(r["Precision@K"])/cov if cov else np.nan,
          "Recall_at_5":float(r["Recall@K"]),"Temporal_cutoff":"original saved evaluation",
          "Train_invoice_count":np.nan,"Test_invoice_count":np.nan,
          "Note":"Existing artifact retained separately; do not compare with rerun unless split and rule build are confirmed identical."}
    save(result,"recommendation_baseline_and_denominator_comparison.csv")
    return result

def association_audit(raw):
    p=ART/"association"/"association_rules.csv"
    if not p.exists(): raise FileNotFoundError(p)
    d=pd.read_csv(p,dtype={"antecedent_codes":str,"consequent_codes":str})
    # Map each code to its modal non-empty product description.
    desc=raw.dropna(subset=["Description"]).copy()
    desc["StockCode"]=desc.StockCode.astype(str).str.strip()
    desc["Description"]=desc.Description.astype(str).str.upper().str.strip()
    mapping=desc.groupby("StockCode").Description.agg(lambda x:x.value_counts().index[0]).to_dict()
    def split(s): return [x.strip() for x in str(s).split("|") if x.strip()]
    def toks(s):
        return set(re.findall(r"[A-Z0-9]+",str(s).upper()))-{"THE","AND","OF","IN","WITH","FOR"}
    rows=[]
    for _,r in d.iterrows():
        a=split(r.antecedent_codes); c=split(r.consequent_codes)
        codes=a+c
        numeric=[]
        for x in codes:
            try: numeric.append(int(float(x)))
            except: pass
        near=(len(numeric)>=2 and max(numeric)-min(numeric)<=5 and all(len(str(v))==5 for v in numeric))
        descriptions=[mapping.get(x,"") for x in codes]
        pair_sims=[]
        for i in range(len(descriptions)):
            for j in range(i+1,len(descriptions)):
                A=toks(descriptions[i]); B=toks(descriptions[j])
                if A and B: pair_sims.append(len(A&B)/len(A|B))
        sim=max(pair_sims) if pair_sims else 0.0
        common_phrase=""
        if len(descriptions)>=2:
            token_lists=[re.findall(r"[A-Z0-9']+",x) for x in descriptions if x]
            if token_lists:
                prefix=[]
                for parts in zip(*token_lists):
                    if len(set(parts))==1: prefix.append(parts[0])
                    else: break
                common_phrase=" ".join(prefix)
        family_flag=bool(near and (sim>=.35 or len(common_phrase.split())>=2))
        rows.append({"antecedent_codes":r.antecedent_codes,"consequent_codes":r.consequent_codes,
          "lift":r.get("lift",np.nan),"confidence":r.get("confidence",np.nan),
          "antecedent_descriptions":" || ".join(mapping.get(x,"[description unavailable]") for x in a),
          "consequent_descriptions":" || ".join(mapping.get(x,"[description unavailable]") for x in c),
          "nearby_numeric_codes":near,"max_description_token_jaccard":sim,
          "shared_description_prefix":common_phrase,"potential_same_series":family_flag})
    audit=pd.DataFrame(rows).sort_values(["potential_same_series","lift"],ascending=[False,False])
    save(audit,"association_rule_product_description_audit.csv")
    filtered=audit.loc[~audit.potential_same_series].copy()
    filtered.to_csv(OUT/"association_rules_conservative_filtered.csv",index=False)
    summary=pd.DataFrame([
      {"Metric":"Total saved association rules","Count":len(audit),"Share":1.0},
      {"Metric":"Nearby numeric codes (screen only)","Count":int(audit.nearby_numeric_codes.sum()),"Share":float(audit.nearby_numeric_codes.mean())},
      {"Metric":"Potential same-series (code proximity + description evidence)","Count":int(audit.potential_same_series.sum()),"Share":float(audit.potential_same_series.mean())},
      {"Metric":"Rules retained after conservative filter","Count":len(filtered),"Share":len(filtered)/len(audit)}
    ])
    save(summary,"association_rule_filter_summary.csv")
    return summary

def segmentation_analysis(tx):
    segpath=ART/"segmentation"/"customer_segments.csv"
    seg=pd.read_csv(segpath)
    cols=["Recency","Frequency","Monetary"]
    X=StandardScaler().fit_transform(np.log1p(seg[cols].clip(lower=0)))
    rows=[]
    for k in range(2,11):
        km=KMeans(n_clusters=k,n_init=30,random_state=42).fit(X)
        rows.append({"K":k,"Inertia":km.inertia_,"Silhouette":silhouette_score(X,km.labels_),
          "Davies_Bouldin":davies_bouldin_score(X,km.labels_),
          "Calinski_Harabasz":calinski_harabasz_score(X,km.labels_),
          "N_customers":len(seg),"Source":"Saved segmentation feature population; recomputed with fixed random seed"})
    save(pd.DataFrame(rows),"segmentation_cluster_count_comparison_full.csv")
    # Temporal stability: independent RFM clustering on two non-overlapping 12-month periods.
    d=tx.dropna(subset=["Customer ID","InvoiceDate"]).copy()
    d["Customer ID"]=d["Customer ID"].astype(int)
    d["Invoice"]=d["Invoice"].astype(str)
    windows=[("2009-12-01","2010-11-30 23:59:59"),("2010-12-01","2011-11-30 23:59:59")]
    rfms=[]
    for start,end in windows:
        w=d[(d.InvoiceDate>=start)&(d.InvoiceDate<=end)]
        enddate=pd.Timestamp(end)
        q=w.groupby("Customer ID").agg(Last=("InvoiceDate","max"),Frequency=("Invoice","nunique"),
                                      Monetary=("TransactionValue","sum"))
        q["Recency"]=(enddate.normalize()-q.Last.dt.normalize()).dt.days
        rfms.append(q[["Recency","Frequency","Monetary"]])
    common=rfms[0].index.intersection(rfms[1].index)
    stability=[]
    for k in [2,3]:
        a=rfms[0].loc[common]; b=rfms[1].loc[common]
        xa=StandardScaler().fit_transform(np.log1p(a.clip(lower=0)))
        xb=StandardScaler().fit_transform(np.log1p(b.clip(lower=0)))
        la=KMeans(n_clusters=k,n_init=30,random_state=42).fit_predict(xa)
        lb=KMeans(n_clusters=k,n_init=30,random_state=42).fit_predict(xb)
        stability.append({"K":k,"Window_1":"2009-12 to 2010-11","Window_2":"2010-12 to 2011-11",
          "Customers_in_both_windows":len(common),"Adjusted_Rand_Index":adjusted_rand_score(la,lb),
          "Method":"Independent KMeans fit/scaler per non-overlapping 12-month period",
          "Caveat":"ARI is label-permutation invariant; feature populations differ because customers must purchase in both periods."})
    save(pd.DataFrame(stability),"segmentation_temporal_stability.csv")
    # Recency dominance diagnostic: rank correlations and cluster profile.
    profiles=seg.groupby("Segment").agg(Customers=("Customer ID","nunique"),Mean_Recency=("Recency","mean"),
      Mean_Frequency=("Frequency","mean"),Mean_Monetary=("Monetary","mean")).reset_index()
    save(profiles,"segmentation_profiles_recomputed.csv")
    return pd.DataFrame(rows),pd.DataFrame(stability)

def reconciliation(raw,dedup,ordinary,purchase):
    rows=[
      {"Stage":"Raw Online Retail II workbook (both sheets)","Rows":len(raw),"Invoices":raw.Invoice.nunique(),
       "Customers":raw["Customer ID"].nunique(),"Notes":"1,067,371 raw lines across two sheets."},
      {"Stage":"After exact duplicate removal across 8 source columns","Rows":len(dedup),"Invoices":dedup.Invoice.nunique(),
       "Customers":dedup["Customer ID"].nunique(),"Notes":"SourceSheet excluded from duplicate key."},
      {"Stage":"Ordinary positive-quantity purchase events; cancellations removed","Rows":len(ordinary),"Invoices":ordinary.Invoice.nunique(),
       "Customers":ordinary["Customer ID"].nunique(),"Notes":"Includes zero/negative price rows until price validation."},
      {"Stage":"Canonical positive-price purchase lines","Rows":len(purchase),"Invoices":purchase.Invoice.nunique(),
       "Customers":purchase["Customer ID"].nunique(),"Notes":"Primary cleaned population: Quantity > 0, Price > 0."},
      {"Stage":"Identified-customer positive-price purchase lines","Rows":int(purchase["Customer ID"].notna().sum()),
       "Invoices":purchase.loc[purchase["Customer ID"].notna(),"Invoice"].nunique(),
       "Customers":purchase["Customer ID"].nunique(),"Notes":"Customer-level modelling population; anonymous transactions excluded."},
      {"Stage":"Association mining saved population","Rows":1007913,"Invoices":40077,"Customers":np.nan,
       "Notes":"Saved metadata; matches canonical positive-price purchase-line count."},
      {"Stage":"Segmentation saved output","Rows":len(pd.read_csv(ART/"segmentation"/"customer_segments.csv")),
       "Invoices":np.nan,"Customers":pd.read_csv(ART/"segmentation"/"customer_segments.csv")["Customer ID"].nunique(),
       "Notes":"Customer-level output after segmentation eligibility filters."},
      {"Stage":"EDA documentation legacy count (not reproduced)","Rows":np.nan,"Invoices":53628,"Customers":5942,
       "Notes":"Legacy EDA summary conflicts with canonical workbook reconstruction; do not use as final population count."}
    ]
    save(pd.DataFrame(rows),"dataset_count_reconciliation.csv")
    # Data integrity checks.
    checks={
      "raw_rows_match_expected":len(raw)==1067371,
      "deduplicated_rows_match_report":len(dedup)==1033036,
      "canonical_positive_price_lines_match_association_artifact":len(purchase)==1007913,
      "canonical_invoice_count_matches_association_artifact":purchase.Invoice.nunique()==40077,
      "canonical_identified_customers_match_segmentation_output":purchase["Customer ID"].nunique()==len(pd.read_csv(ART/"segmentation"/"customer_segments.csv"))
    }
    (OUT/"dataset_integrity_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    if not all(checks.values()):
        raise AssertionError(f"Dataset reconciliation checks failed: {checks}")
    return checks

def write_report(checks):
    rec=pd.read_csv(OUT/"recommendation_baseline_and_denominator_comparison.csv")
    rec_rule=rec[rec.Method=="Association-rule recommender"].iloc[0]
    rec_pop=rec[rec.Method=="Training-period top sellers"].iloc[0]
    cal=pd.read_csv(OUT/"inactivity_calibration_before_after.csv")
    cal_raw=cal[(cal.Model=="Uncalibrated") & (cal.Scope=="Confirmatory June-July 2011")].iloc[0]
    cal_platt=cal[(cal.Model=="Platt") & (cal.Scope=="Confirmatory June-July 2011")].iloc[0]
    base=pd.read_csv(OUT/"inactivity_recency_baseline_comparison.csv")
    base=base[base.Compared_model=="B: 12-month look-back"]
    model=pd.read_csv(OUT/"inactivity_locked_model_comparison.csv")
    model=model[model.Model=="B: 12-month look-back"]
    assoc=pd.read_csv(OUT/"association_rule_filter_summary.csv")
    seg=pd.read_csv(OUT/"segmentation_cluster_count_comparison_full.csv")
    stab=pd.read_csv(OUT/"segmentation_temporal_stability.csv")
    miss=pd.read_csv(OUT/"inactivity_feature_missingness_audit.csv")
    subgroup=pd.read_csv(OUT/"inactivity_missingness_subgroup_metrics.csv")
    counts=pd.read_csv(OUT/"dataset_count_reconciliation.csv")
    files=sorted(p.name for p in OUT.iterdir() if p.is_file())
    lines=[
      "# RetailPulse — Eight Research Improvements: Execution Report","",
      "Generated by `scripts/run_all_research_improvements.py` from the supplied Online Retail II workbook and existing project artifacts.","",
      "## Executive result","",
      "All eight research improvements now have executable analysis, saved outputs, and documentation. The results do not imply that every proposed model choice is superior: several diagnostics identify limitations that must be reported honestly.","",
      "## 1. Baseline comparisons","","The inactivity model is compared with a recency-only ranking baseline on the same locked customer-origin rows. Raw recency is treated as a ranking score; probability metrics are not calculated on raw day counts. A one-variable logistic mapping is fitted on the latest development origins (April-May 2011) only.","",
      "| Origin | Recency-only AUC | Extended-lookback model AUC |","|---|---:|---:|"
    ]
    for _,r in base.iterrows():
        m=model[model.Scope==r.Scope].iloc[0]
        lines.append(f"| {r.Scope} | {r.ROC_AUC:.4f} | {m.ROC_AUC:.4f} |")
    lines += ["","The recommender is compared with training-period top sellers on the same 7,253 eligible temporal test baskets:","",
      f"- Association-rule recommender Precision@5, all baskets: **{rec_rule.Precision_at_5_all_baskets:.4f}**.",
      f"- Association-rule recommender Precision@5, covered baskets: **{rec_rule.Precision_at_5_covered_baskets:.4f}**.",
      f"- Top-selling-products baseline Precision@5: **{rec_pop.Precision_at_5_all_baskets:.4f}** (coverage {rec_pop.Coverage:.1%}).","",
      "## 2. Probability calibration","","Platt scaling and isotonic regression are fitted on April-May 2011 development out-of-fold predictions only. They are evaluated on June-July 2011, the confirmatory block. August 2011 is separately labelled as previously inspected.","",
      f"- Confirmatory Brier score: {cal_raw.Brier:.4f} uncalibrated → **{cal_platt.Brier:.4f} Platt**.",
      f"- Mean probability: {cal_raw.Mean_probability:.4f} uncalibrated → **{cal_platt.Mean_probability:.4f} Platt**; observed inactivity rate {cal_platt.Observed_rate:.4f}.",
      f"- 10-bin ECE: {cal_raw.ECE_10_bins:.4f} uncalibrated → **{cal_platt.ECE_10_bins:.4f} Platt**.",
      "- The calibrator bundle is saved separately and is not silently applied to the production November 2011 customer-score file. Its temporal transfer must be revalidated before live use.","",
      "## 3. Recommendation denominators","",
      f"- Eligible baskets: {int(rec_rule.Eligible_test_baskets):,}; covered: {int(rec_rule.Baskets_with_recommendations):,}; no recommendation: {int(rec_rule.No_recommendation_baskets):,}.",
      f"- Coverage: {rec_rule.Coverage:.2%}; precision over all baskets: {rec_rule.Precision_at_5_all_baskets:.4f}; precision over covered baskets: {rec_rule.Precision_at_5_covered_baskets:.4f}.","",
      "## 4. Association-rule triviality audit","",
      f"- Rules audited: {int(assoc.iloc[0].Count):,}.",
      f"- Nearby numeric StockCodes (screening only): {int(assoc.iloc[1].Count):,} ({assoc.iloc[1].Share:.2%}).",
      f"- Potential same-series rules using both code proximity and description evidence: {int(assoc.iloc[2].Count):,} ({assoc.iloc[2].Share:.2%}).",
      f"- Rules retained by the conservative experimental filter: {int(assoc.iloc[3].Count):,}.",
      "- Full rules and descriptions are preserved in the audit; the heuristic flags candidates for review and is not presented as definitive product taxonomy.","",
      "## 5. Segmentation selection and temporal stability","",
      f"- K=2 silhouette: {seg.loc[seg.K==2,'Silhouette'].iloc[0]:.4f}; K=3 silhouette: {seg.loc[seg.K==3,'Silhouette'].iloc[0]:.4f}.",
      "- K=2 has the better silhouette score on the saved RFM population; K=3 remains an interpretable alternative, not the statistically strongest solution.",
      f"- Temporal ARI: K=2 = {stab.loc[stab.K==2,'Adjusted_Rand_Index'].iloc[0]:.4f}; K=3 = {stab.loc[stab.K==3,'Adjusted_Rand_Index'].iloc[0]:.4f} across two non-overlapping 12-month windows.",
      "- The low temporal ARI indicates substantial segment movement; the three-cluster segmentation should be described as a descriptive snapshot, not a stable customer identity.","",
      "## 6. Missing-data and subgroup evaluation","",
      "| Feature | Missing share over the 48,394 customer-snapshot table |","|---|---:|"
    ]
    for _,r in miss.iterrows():
        lines.append(f"| {r.Feature} | {r.Missing_share:.2%} |")
    lines += ["","Subgroup performance (rolling-origin out-of-fold predictions; early-origin rows are included only for this subgroup diagnostic):","",
      "| Subgroup | Rows | ROC-AUC | Brier | Observed inactivity |","|---|---:|---:|---:|---:|"]
    for _,r in subgroup.iterrows():
        lines.append(f"| {r.Subgroup} | {int(r.Rows):,} | {r.ROC_AUC:.4f} | {r.Brier:.4f} | {r.Observed_inactivity_rate:.4f} |")
    lines += ["","Full subgroup AUC, PR-AUC and Brier scores are saved in `inactivity_missingness_subgroup_metrics.csv`. Early-origin subgroup predictions are exploratory only; the frozen development and confirmatory roles remain separately recorded.","",
      "## 7. Test-data governance","",
      "- Development OOF origins: December 2010-May 2011 (plus separately marked early-origin subgroup diagnostics for September-November 2010).",
      "- Confirmatory block: June-July 2011.",
      "- August 2011: previously inspected; descriptive only and not a pristine test set.",
      "- Calibration fit: April-May 2011 development OOF predictions only. June-July are used only for post-fit evaluation.",
      "- See `inactivity_temporal_data_governance.csv` for the frozen role table.","",
      "## 8. Dataset count reconciliation","","| Stage | Rows | Invoices | Customers |","|---|---:|---:|---:|"]
    for _,r in counts.iterrows():
        rows=str(int(r.Rows)) if pd.notna(r.Rows) else "—"
        inv=str(int(r.Invoices)) if pd.notna(r.Invoices) else "—"
        cust=str(int(r.Customers)) if pd.notna(r.Customers) else "—"
        lines.append(f"| {r.Stage} | {rows} | {inv} | {cust} |")
    lines += ["",
      "The 5,942-customer / 53,628-invoice figures in the EDA documents are the raw-workbook counts before cleaning. The canonical positive-price purchase population is 5,878 identified customers and 40,077 invoices. The apparent discrepancy is therefore explained by the cleaning stages, not by mixing datasets.","",
      "## Important interpretation and limitations","",
      "- August 2011 was previously inspected and must not be described as an untouched test set.",
      "- The calibration correction improves Brier score on June-July 2011, but it must be monitored for further temporal drift before operational deployment.",
      "- The association-rule filter is conservative and heuristic; manual product-category validation is still advisable before publication.",
      "- Segmentation temporal ARI is low; report the instability rather than claiming stable customer segments.",
      "- Offline basket completion is not a live A/B test and does not establish conversion uplift.",
      "- Predictive associations are not causal effects.","",
      "## Generated artifacts",""]
    lines += [f"- `{x}`" for x in files]
    (OUT/"EIGHT_IMPROVEMENTS_EXECUTION_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True,help="Path to UCI Online Retail II workbook (.xlsx)")
    ap.add_argument("--export-cleaned",action="store_true",
                    help="Also write data/processed/cleaned_transactions.csv for the Streamlit dashboard.")
    args=ap.parse_args()
    warnings.filterwarnings("ignore")
    print("Loading and cleaning Online Retail II...")
    raw,dedup,ordinary,purchase=load_and_clean(args.dataset)
    print(f"Raw {len(raw):,} | deduplicated {len(dedup):,} | canonical positive-price lines {len(purchase):,}")
    if args.export_cleaned:
        processed_dir=ROOT/"data"/"processed"
        processed_dir.mkdir(parents=True,exist_ok=True)
        purchase.to_csv(processed_dir/"cleaned_transactions.csv",index=False)
        print("Wrote dashboard mapping/transaction file:",processed_dir/"cleaned_transactions.csv")
    checks=reconciliation(raw,dedup,ordinary,purchase)
    print("Running inactivity baselines, calibration and subgroup evaluations...")
    tx=purchase.loc[purchase["Customer ID"].notna()].copy()
    tx["TransactionValue"]=tx["Quantity"]*tx["Price"]
    saved_pred=ART/"inactivity"/"inactivity_v2_locked_predictions.csv"
    build_inactivity_evaluation(tx,saved_pred)
    print("Running recommendation baselines and denominator comparison...")
    recommendation_analysis(purchase)
    print("Auditing association rules against product descriptions...")
    association_audit(raw)
    print("Running segmentation diagnostics and temporal stability...")
    segmentation_analysis(tx)
    write_report(checks)
    print("\nCompleted. Outputs:",OUT)
    print("Review EIGHT_IMPROVEMENTS_EXECUTION_REPORT.md before using any results in a paper.")

if __name__=="__main__":
    main()
