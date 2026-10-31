"""Part B: RFM segmentation and a leakage-aware lapse model."""
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.base import clone
ROOT=Path(__file__).resolve().parents[1]; CLEAN=ROOT/"data/clean"; OUT=ROOT/"outputs"
SNAPSHOT=pd.Timestamp("2025-10-01")
def load():
    c=pd.read_csv(CLEAN/"customers.csv"); o=pd.read_csv(CLEAN/"orders.csv")
    o["order_ts"]=pd.to_datetime(o.order_ts)
    return c,o
def rfm_and_segments(k=4):
    c,o=load(); done=o[o.status.eq("completed")].copy()
    rfm=done.groupby("customer_id").agg(last_order=("order_ts","max"),frequency=("order_id","nunique"),monetary=("order_amount","sum")).reset_index()
    rfm["recency"]=(SNAPSHOT-rfm.last_order).dt.days; rfm=rfm.drop(columns="last_order")
    scaler=StandardScaler(); X=scaler.fit_transform(rfm[["recency","frequency","monetary"]])
    diagnostics=[]
    from sklearn.metrics import silhouette_score
    for n in range(2,9):
        km=KMeans(n_clusters=n,n_init=10,random_state=42).fit(X)
        diagnostics.append({"k":n,"inertia":km.inertia_,"silhouette":silhouette_score(X,km.labels_)})
    km=KMeans(n_clusters=k,n_init=10,random_state=42); rfm["segment_id"]=km.fit_predict(X)
    profile=rfm.groupby("segment_id").agg(size=("customer_id","size"),recency=("recency","mean"),frequency=("frequency","mean"),monetary=("monetary","mean")).reset_index()
    remaining=profile.set_index("segment_id").drop(
        index=profile.loc[profile.monetary.idxmax(),"segment_id"]
    )
    premium_id=profile.loc[profile.monetary.idxmax(),"segment_id"]
    frequent_id=remaining.frequency.idxmax()
    remaining=remaining.drop(index=frequent_id)
    dormant_id=remaining.recency.idxmax()
    name_map={
        premium_id:"Premium",
        frequent_id:"Frequent",
        dormant_id:"Dormant",
    }
    remaining_ids=[segment_id for segment_id in profile.segment_id if segment_id not in name_map]
    for index,segment_id in enumerate(remaining_ids,1):
        suffix=f" {index}" if len(remaining_ids)>1 else ""
        name_map[segment_id]=f"Occasional{suffix}"
    rfm["segment_name"]=rfm.segment_id.map(name_map); rfm["is_premium"]=rfm.segment_id.eq(premium_id)
    OUT.mkdir(exist_ok=True)
    rfm[["customer_id","recency","frequency","monetary","segment_id","segment_name","is_premium"]].to_csv(OUT/"customer_segments.csv",index=False)
    pca=PCA(n_components=2).fit(X)
    return rfm,pd.DataFrame(diagnostics),profile,pca.explained_variance_ratio_
def build_lapse_model():
    c,o=load(); cutoff=pd.Timestamp("2025-09-01"); end=pd.Timestamp("2025-10-01")
    hist=o[(o.order_ts<cutoff)].copy(); done=hist[hist.status.eq("completed")]
    cohort=done.groupby("customer_id").agg(last_order=("order_ts","max"),frequency=("order_id","nunique"),monetary=("order_amount","sum"),
      avg_discount=("discount_pct","mean"),avg_delivery=("delivery_minutes","mean")).reset_index()
    cohort["recency"]=(cutoff-cohort.last_order).dt.days; cohort=cohort.drop(columns="last_order")
    all_final=hist[hist.status.isin(["completed","cancelled"])].groupby("customer_id").status.agg(total="count",cancelled=lambda x:(x=="cancelled").sum()).reset_index()
    cohort=cohort.merge(all_final,on="customer_id",how="left"); cohort["cancellation_rate"]=cohort.cancelled/cohort.total
    cats=done.merge(pd.read_csv(CLEAN/"products.csv")[["product_id","category"]],on="product_id").groupby(["customer_id","category"]).size().rename("n").reset_index()
    fav=cats.sort_values(["customer_id","n","category"],ascending=[True,False,True]).drop_duplicates("customer_id").rename(columns={"category":"favourite_category"})[["customer_id","favourite_category"]]
    cohort=cohort.merge(fav,on="customer_id",how="left").merge(c,on="customer_id",how="left")
    cohort["signup_date"]=pd.to_datetime(cohort.signup_date); cohort["tenure_days"]=(cutoff-cohort.signup_date).dt.days
    post=o[(o.order_ts>=cutoff)&(o.order_ts<end)&o.status.eq("completed")].groupby("customer_id").size()
    cohort["lapsed"]=(~cohort.customer_id.isin(post.index)).astype(int)
    cohort=cohort.drop(columns=["signup_date","total","cancelled"])
    features=["recency","frequency","monetary","avg_discount","cancellation_rate","avg_delivery","favourite_category","city","age","gender","tenure_days"]
    X=cohort[features].copy(); y=cohort.lapsed
    numeric=["recency","frequency","monetary","avg_discount","cancellation_rate","avg_delivery","age","tenure_days"]
    categorical=["favourite_category","city","gender"]
    if X.isna().any().any():
        missing=X.columns[X.isna().any()].tolist()
        raise ValueError(f"Lapse features contain missing values: {missing}")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    prep=ColumnTransformer([("num",StandardScaler(),numeric),("cat",OneHotEncoder(drop="first",handle_unknown="ignore"),categorical)])
    models={}
    for name,est,grid in [
      ("logistic",LogisticRegression(max_iter=1000),{"model__C":[.01,.1,1,10,100]}),
      ("forest",RandomForestClassifier(n_estimators=100,random_state=42),{"model__max_depth":[3,5,8],"model__min_samples_leaf":[5,10,20]})]:
        pipe=Pipeline([("prep",prep),("model",est)])
        search=GridSearchCV(pipe,grid,cv=5,scoring="f1",n_jobs=-1); search.fit(X_train,y_train)
        pred=search.predict(X_test)
        models[name]={"best_params":search.best_params_,"cv_f1":search.best_score_,"precision":precision_score(y_test,pred,zero_division=0),
          "recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),"confusion_matrix":confusion_matrix(y_test,pred).tolist(),"search":search}
    best_logistic=models["logistic"]["search"].best_estimator_
    best_forest=models["forest"]["search"].best_estimator_
    selected=models["logistic"]["search"]
    baseline_prediction=np.zeros(len(y_test),dtype=int)
    balanced_metrics={}
    minority_class=int(y_train.value_counts().idxmin())
    minority_count=int(y_train.eq(minority_class).sum())
    majority_count=int(y_train.eq(1-minority_class).sum())
    additional=majority_count-minority_count
    X_balanced=pd.concat([
        X_train,
        X_train.loc[y_train[y_train.eq(minority_class)].index].sample(
            n=additional,replace=True,random_state=42
        ),
    ])
    y_balanced=pd.concat([
        y_train,
        y_train.loc[y_train.eq(minority_class)].sample(
            n=additional,replace=True,random_state=42
        ),
    ])
    balanced_logistic=clone(best_logistic).fit(X_balanced,y_balanced)
    balanced_prediction=balanced_logistic.predict(X_test)
    balanced_metrics={
        "train_class_counts":{str(k):int(v) for k,v in y_balanced.value_counts().sort_index().items()},
        "precision":precision_score(y_test,balanced_prediction,zero_division=0),
        "recall":recall_score(y_test,balanced_prediction,zero_division=0),
        "f1":f1_score(y_test,balanced_prediction,zero_division=0),
    }
    OUT.mkdir(exist_ok=True); joblib.dump(selected,OUT/"lapse_model.joblib")
    assert np.array_equal(selected.predict(X_test),joblib.load(OUT/"lapse_model.joblib").predict(X_test))
    # Premium list is illustrative: the flag uses whole-window RFM and is post-hoc.
    segments,_,_,_=rfm_and_segments()
    premium=set(segments.loc[segments.is_premium,"customer_id"])
    probs=selected.predict_proba(X_test)[:,list(selected.classes_).index(1)]
    ret=pd.DataFrame({"customer_id":cohort.loc[X_test.index,"customer_id"].values,
      "lapse_probability":probs,"actually_lapsed":y_test.values})
    ret=ret[ret.customer_id.isin(premium)].sort_values("lapse_probability",ascending=False).head(10)
    ret.to_csv(OUT/"retention_list.csv",index=False)
    forest=best_forest.named_steps["model"]
    feature_names=best_forest.named_steps["prep"].get_feature_names_out()
    importance=pd.Series(forest.feature_importances_,index=feature_names).sort_values(ascending=False)
    logit_features=best_logistic.named_steps["prep"].get_feature_names_out()
    logit_coefficients=pd.Series(best_logistic.named_steps["model"].coef_[0],index=logit_features)
    train_frequency_mean=float(best_logistic.named_steps["prep"].named_transformers_["num"].mean_[numeric.index("frequency")])
    return {
      "cohort_rows":len(cohort),"lapsed":int(y.sum()),"lapsed_rate":float(y.mean()),
      "train_rows":len(X_train),"test_rows":len(X_test),
      "baseline_accuracy":accuracy_score(y_test,baseline_prediction),
      "baseline_f1":f1_score(y_test,baseline_prediction,zero_division=0),
      "models":{k:{a:b for a,b in v.items() if a!="search"} for k,v in models.items()},
      "class_balance_logistic":balanced_metrics,
      "forest_feature_importance":[{"feature":str(name),"importance":float(value)} for name,value in importance.head(5).items()],
      "logistic_coefficients":{name:float(logit_coefficients[name]) for name in [
        "num__recency","num__frequency","num__monetary"
      ]},
      "frequency_scaler_mean":train_frequency_mean,
      "frequency_cohort_mean":float(cohort.frequency.mean()),
      "retention_list":ret,
      "retention_true_lapsed":int(ret.actually_lapsed.sum()),
    }
