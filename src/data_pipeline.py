"""Part A: audit, clean, validate and load data into SQLite."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"; CLEAN=ROOT/"data"/"clean"; DB=ROOT/"data"/"instamart.db"
def audit():
    for name in ["customers","products","orders"]:
        df=pd.read_csv(RAW/f"{name}.csv",keep_default_na=False)
        print(f"\n{name}: shape={df.shape}")
        print(df.dtypes)
        print(pd.DataFrame({"missing_count":df.isna().sum(),"missing_pct":(df.isna().mean()*100).round(2)}))
    orders=pd.read_csv(RAW/"orders.csv",keep_default_na=False)
    cust=pd.read_csv(RAW/"customers.csv",keep_default_na=False)
    print("\nOrders audit:",{"rows":len(orders),"exact_duplicates":int(orders.duplicated().sum()),
      "pending":int(orders.status.eq("pending").sum()),"distinct_order_id":int(orders.order_id.nunique()),
      "bad_amount":int(pd.to_numeric(orders.order_amount,errors="coerce").isna().sum()),
      "negative_delivery":int(pd.to_numeric(orders.delivery_minutes,errors="coerce").lt(0).sum()),
      "blank_delivery":int(orders.delivery_minutes.astype(str).str.strip().eq("").sum())})
    print("Customer audit:",{"blank_age":int(cust.age.astype(str).str.strip().eq("").sum()),
      "age_outside_10_100":int(pd.to_numeric(cust.age,errors="coerce").pipe(lambda x: ((x<10)|(x>100)).sum())),
      "raw_city_spellings":int(cust.city.nunique()),"raw_gender_spellings":int(cust.loc[cust.gender.astype(str).str.strip().ne(""),"gender"].nunique()),
      "blank_gender":int(cust.gender.astype(str).str.strip().eq("").sum())})
def clean_data():
    CLEAN.mkdir(parents=True,exist_ok=True)
    cust=pd.read_csv(RAW/"customers.csv",keep_default_na=False)
    prod=pd.read_csv(RAW/"products.csv")
    orders=pd.read_csv(RAW/"orders.csv",keep_default_na=False)
    cust["city"]=cust.city.astype(str).str.strip().str.title()
    cust["gender"]=cust.gender.astype(str).str.strip().str.lower().map({"male":"male","m":"male","female":"female","f":"female"}).fillna("unknown")
    cust["age"]=pd.to_numeric(cust.age,errors="coerce")
    cust.loc[~cust.age.between(10,100),"age"]=np.nan
    cust["age"]=cust.age.fillna(cust.groupby("city").age.transform("median")).round().astype(int)
    cust["signup_date"]=pd.to_datetime(cust.signup_date).dt.strftime("%Y-%m-%d")
    orders["order_ts"]=pd.to_datetime(orders.order_ts)
    orders=orders.drop_duplicates()
    # Latest timestamp wins when the same order appears as pending and final.
    orders=orders.sort_values(["order_id","order_ts"]).drop_duplicates("order_id",keep="last").copy()
    orders["quantity"]=pd.to_numeric(orders.quantity,errors="coerce")
    orders["discount_pct"]=pd.to_numeric(orders.discount_pct,errors="coerce").fillna(0)
    orders["order_amount"]=pd.to_numeric(orders.order_amount,errors="coerce")
    orders["delivery_minutes"]=pd.to_numeric(orders.delivery_minutes,errors="coerce")
    orders.loc[orders.delivery_minutes<0,"delivery_minutes"]=np.nan
    price=prod.set_index("product_id").price_inr
    missing=orders.order_amount.isna()
    orders.loc[missing,"order_amount"]=(orders.loc[missing,"product_id"].map(price)*orders.loc[missing,"quantity"]*(1-orders.loc[missing,"discount_pct"]/100)).round(2)
    orders["category"]=orders.product_id.map(prod.set_index("product_id").category)
    completed=orders.status.eq("completed")
    med=orders.loc[completed].groupby("category").delivery_minutes.transform("median")
    # Category medians mapped to all completed rows, including missing values.
    cat_median=orders.loc[completed].groupby("category").delivery_minutes.median()
    fillmask=completed & orders.delivery_minutes.isna()
    orders.loc[fillmask,"delivery_minutes"]=orders.loc[fillmask,"category"].map(cat_median)
    orders["delivery_minutes"]=orders.delivery_minutes.round().astype("Int64")
    orders=orders.drop(columns=["category"])
    orders["order_ts"]=orders.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    for name,df in [("customers",cust),("products",prod),("orders",orders)]:
        df.to_csv(CLEAN/f"{name}.csv",index=False)
    DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as con:
        for name,df in [("customers",cust),("products",prod),("orders",orders)]:
            df.to_sql(name,con,if_exists="replace",index=False)
    return {"customers":cust,"products":prod,"orders":orders}
def validate():
    c=pd.read_csv(CLEAN/"customers.csv"); o=pd.read_csv(CLEAN/"orders.csv",keep_default_na=False)
    assert c.city.nunique()==6
    assert set(c.gender.unique())<={"male","female","unknown"}
    assert c.age.notna().all() and c.age.between(18,60).all()
    assert o.order_id.is_unique
    assert not o.status.eq("pending").any()
    assert pd.to_numeric(o.order_amount,errors="coerce").notna().all()
    d=pd.to_numeric(o.loc[o.status.eq("completed"),"delivery_minutes"],errors="coerce")
    assert d.notna().all()
    assert pd.to_numeric(o.loc[o.status.eq("cancelled"),"delivery_minutes"],errors="coerce").isna().all()
    with sqlite3.connect(DB) as con:
        names={r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        assert {"customers","products","orders"}<=names
    print("ALL CHECKPOINTS PASSED")
if __name__=="__main__":
    audit(); clean_data(); validate()
