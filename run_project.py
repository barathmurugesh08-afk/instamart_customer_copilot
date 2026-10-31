"""Reproduce the data, modelling, triage, retrieval, and refund-agent workflow."""
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

from src.data_pipeline import audit, clean_data, validate
from src.customer_models import build_lapse_model, rfm_and_segments
from src.triage import FEW, ZERO, run_pipeline
from src.policy_rag import PolicyRAG, evaluate
from src.refund_agent import (
    build_agent_scenarios,
    resume_agent,
    run_agent,
)

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"


def _run_sql_queries():
    sql=(ROOT/"sql/queries.sql").read_text(encoding="utf-8")
    sections=re.split(r"(?m)(?=-- Q\d+:)",sql)
    queries=[section.strip() for section in sections if re.match(r"-- Q\d+:",section.strip())]
    with sqlite3.connect(ROOT/"data/instamart.db") as con:
        results=[pd.read_sql_query(query,con) for query in queries]
    if len(results)!=7:
        raise ValueError(f"Expected seven SQL queries, found {len(results)}.")
    for number,result in enumerate(results,1):
        print(f"Q{number}: {len(result)} rows")
        if number in (2,6,7):
            print(result.head(8).to_string(index=False))
    return results


def _report_data_analysis():
    orders=pd.read_csv(ROOT/"data/clean/orders.csv")
    customers=pd.read_csv(ROOT/"data/clean/customers.csv")
    products=pd.read_csv(ROOT/"data/clean/products.csv")
    completed=orders[orders.status.eq("completed")].copy()
    revenue=completed.merge(customers[["customer_id","city"]],on="customer_id").merge(
        products[["product_id","category"]],on="product_id"
    )
    pivot=pd.pivot_table(
        revenue,index="city",columns="category",values="order_amount",
        aggfunc="sum",fill_value=0,margins=True,
    )
    city_summary=completed.merge(customers[["customer_id","city"]],on="customer_id").groupby(
        "city"
    ).order_amount.agg(total="sum",mean="mean",count="count").sort_index()
    amount=completed.order_amount
    q1,q3=amount.quantile([.25,.75])
    iqr=q3-q1
    fences={"lower":q1-1.5*iqr,"upper":q3+1.5*iqr}
    outliers={
        "below":int(amount.lt(fences["lower"]).sum()),
        "above":int(amount.gt(fences["upper"]).sum()),
    }
    correlations=completed[[
        "order_amount","discount_pct","quantity","delivery_minutes"
    ]].corr()
    pairs=correlations.abs().where(~pd.DataFrame(
        [[i==j for j in correlations.columns] for i in correlations.index],
        index=correlations.index,columns=correlations.columns,
    )).stack()
    strongest_pair=pairs.idxmax()
    print("\n=== PART A: SQL and pandas analysis ===")
    queries=_run_sql_queries()
    print("Q2 city/category grand total:",round(float(queries[1].revenue.sum()),2))
    print("Q6 tie rows:",int((queries[5]["spend_rank"]!=queries[5]["row_num"]).sum()))
    print("Q7 days and final cumulative total:",len(queries[6]),round(float(queries[6].cumulative_revenue.iloc[-1]),2))
    print("Pandas pivot grand total:",round(float(pivot.loc["All","All"]),2))
    print("Completed order totals by city:\n",city_summary.round(2).to_string())
    print("Order amount stats:",{
        "mean":float(amount.mean()),"median":float(amount.median()),
        "skew":float(amount.skew()),"q1":float(q1),"q3":float(q3),
        "lower_fence":float(fences["lower"]),"upper_fence":float(fences["upper"]),
        "outliers":outliers,
    })
    print("Correlation matrix:\n",correlations.round(3).to_string())
    print(
        "Largest absolute correlation pair:",strongest_pair,
        float(correlations.loc[strongest_pair[0],strongest_pair[1]]),
    )
    return {
        "queries":queries,
        "pivot":pivot,
        "city_summary":city_summary,
        "amount_stats":{
            "mean":float(amount.mean()),"median":float(amount.median()),
            "skew":float(amount.skew()),"q1":float(q1),"q3":float(q3),
            "lower_fence":float(fences["lower"]),"upper_fence":float(fences["upper"]),
            "outliers":outliers,
        },
        "correlations":correlations,
        "strongest_pair":strongest_pair,
    }


def _append_agent_run(run_id,scenario_id,result):
    row={
        "run_id":run_id,
        "scenario_id":scenario_id,
        "status":result["status"],
        "final_answer":result["final_answer"],
        "llm_calls":result["llm_calls"],
        "tokens_used":result["tokens_used"],
        "steps":result["steps"],
    }
    with (OUT/"agent_runs.jsonl").open("a",encoding="utf-8") as file:
        file.write(json.dumps(row,ensure_ascii=False)+"\n")
    return row


def _refund_amount(result):
    for step in result["steps"]:
        if step["tool"] in {"issue_refund","escalate_to_human"}:
            return float(step["observation"]["refund_inr"])
    return 0.0


def main():
    OUT.mkdir(exist_ok=True)
    print("\n=== PART A: Audit, clean, validate ===")
    audit()
    clean_data()
    validate()
    a_report=_report_data_analysis()

    print("\n=== PART B: Segments and lapse model ===")
    rfm,diagnostics,profile,pca=rfm_and_segments(k=4)
    print("K-means diagnostics:\n",diagnostics.round(3).to_string(index=False))
    print("Segment profile:\n",profile.round(2).to_string(index=False))
    print("RFM rows/recency median/min/max:",len(rfm),rfm.recency.median(),rfm.recency.min(),rfm.recency.max())
    print("PCA explained variance:",pca,"total:",pca.sum())
    model_report=build_lapse_model()
    if model_report["models"]["logistic"]["f1"]<0.55:
        raise AssertionError("The selected lapse model did not meet the minimum test F1.")
    if len(model_report["retention_list"])>10:
        raise AssertionError("The retention list contains more than ten customers.")
    print("Lapse model report:",json.dumps(model_report,default=str,indent=2))

    print("\n=== PART C: Compare prompts and enforce budget ===")
    zero_result,zero_report=run_pipeline(ZERO,100000,OUT/"triage_results.csv")
    few_result,few_report=run_pipeline(FEW,100000,OUT/"triage_results.csv")
    budget_result,budget_report=run_pipeline(ZERO,6000,OUT/"triage_budget_demo.csv")
    if zero_report["classified_tickets"]!=147 or abs(zero_report["accuracy"]-0.8571428571)>0.005:
        raise AssertionError("Zero-shot triage did not match the reference run.")
    if few_report["classified_tickets"]!=147 or few_report["accuracy"]!=1.0:
        raise AssertionError("Few-shot triage did not match the reference run.")
    if zero_report["status_counts"].get("classified_json")!=135 or zero_report["status_counts"].get("classified_keyword")!=12 or zero_report["status_counts"].get("flagged_invalid_output")!=9:
        raise AssertionError("Triage output validation counts did not match the reference run.")
    if budget_report["status_counts"].get("budget_exhausted",0)<1:
        raise AssertionError("The token-budget demonstration did not exhaust its budget.")
    if len(few_result)!=164 or len(budget_result)!=164:
        raise AssertionError("Triage output CSVs must each contain all 164 tickets.")
    print("Zero-shot:",{k:v for k,v in zero_report.items() if k!="confusion_matrix"})
    print("Zero-shot confusion matrix:\n",zero_report["confusion_matrix"].to_string())
    print("Few-shot:",{k:v for k,v in few_report.items() if k!="confusion_matrix"})
    print("Few-shot confusion matrix:\n",few_report["confusion_matrix"].to_string())
    print("Budget demo exhausted tickets:",budget_report["status_counts"].get("budget_exhausted",0))

    print("\n=== PART D: Policy retrieval and refund agent ===")
    rag=PolicyRAG(chunk_size=40,overlap=10)
    policy_report=evaluate(rag)
    no_overlap_report=evaluate(PolicyRAG(chunk_size=40,overlap=0))
    if len(rag.chunks)!=15:
        raise AssertionError(f"Expected 15 policy chunks, found {len(rag.chunks)}.")
    if any(hit["source"]!="refund_policy.txt" for hit in rag.retrieve("refund policy late delivery",3)):
        raise AssertionError("The top refund-policy retrieval results came from unexpected documents.")
    if int(policy_report.retrieved_doc.sum())!=12 or int(policy_report.fact_found.sum())!=12 or int(policy_report.fallback.sum())!=2:
        raise AssertionError("Overlapping policy retrieval did not match the reference evaluation.")
    no_overlap_answerable=no_overlap_report[no_overlap_report.expected_doc.ne("NONE")]
    if no_overlap_answerable.fact_found.all():
        raise AssertionError("The zero-overlap evaluation did not expose a chunk-boundary failure.")
    print("Policy chunks:",len(rag.chunks))
    print("Policy evaluation with overlap:\n",policy_report.to_string(index=False))
    print("Policy coverage with overlap:",{
        "expected_doc":int(policy_report.retrieved_doc.sum()),
        "expected_fact":int(policy_report.fact_found.sum()),
        "fallbacks":int(policy_report.fallback.sum()),
    })
    print("Policy evaluation without overlap:\n",no_overlap_report.to_string(index=False))
    print("No-overlap failures:",no_overlap_report.loc[
        (~no_overlap_report.retrieved_doc)|(~no_overlap_report.fact_found),
        "question",
    ].tolist())

    scenarios_path=build_agent_scenarios()
    scenarios=pd.read_csv(scenarios_path)
    if len(scenarios)!=7:
        raise AssertionError(f"Expected seven agent scenarios, found {len(scenarios)}.")
    (OUT/"agent_runs.jsonl").write_text("",encoding="utf-8")
    (OUT/"refund_ledger.jsonl").write_text("",encoding="utf-8")
    scenario_results=[]
    for _,scenario in scenarios.iterrows():
        result=run_agent(
            scenario.message,scenario_id=scenario.scenario_id,
            run_id=f"scenario-{scenario.scenario_id}",
        )
        expected_status=(
            "FALLBACK_MAX_ITERATIONS"
            if scenario.expected_outcome=="FALLBACK" else scenario.expected_outcome
        )
        if result["status"]!=expected_status:
            raise AssertionError(
                f"{scenario.scenario_id} status {result['status']} != {expected_status}."
            )
        actual_refund=_refund_amount(result)
        if abs(actual_refund-float(scenario.expected_refund_inr))>0.01:
            raise AssertionError(f"{scenario.scenario_id} refund amount is incorrect.")
        if scenario.scenario_id=="S1" and result["llm_calls"]!=5:
            raise AssertionError("S1 must complete in five model calls.")
        if scenario.scenario_id=="S7" and result["llm_calls"]!=6:
            raise AssertionError("S7 must reach its six-call iteration limit.")
        _append_agent_run(f"scenario-{scenario.scenario_id}",scenario.scenario_id,result)
        scenario_results.append({
            "scenario_id":scenario.scenario_id,
            "status":result["status"],
            "refund_inr":actual_refund,
            "expected_outcome":scenario.expected_outcome,
            "expected_refund_inr":float(scenario.expected_refund_inr),
            "llm_calls":result["llm_calls"],
            "tokens_used":result["tokens_used"],
        })
    print("Seven scenarios:\n",pd.DataFrame(scenario_results).to_string(index=False))

    s1=scenarios.loc[scenarios.scenario_id.eq("S1")].iloc[0]
    token_demo=run_agent(s1.message,scenario_id="S1-token-limit",token_limit=200,run_id="S1-token-limit")
    timeout_demo=run_agent(s1.message,scenario_id="S1-timeout",time_limit_seconds=0,run_id="S1-timeout")
    _append_agent_run("S1-token-limit","S1-token-limit",token_demo)
    _append_agent_run("S1-timeout","S1-timeout",timeout_demo)
    if token_demo["status"]!="FALLBACK_TOKEN_LIMIT" or timeout_demo["status"]!="FALLBACK_TIMEOUT":
        raise AssertionError("A stop-safely demonstration did not return its expected fallback.")
    print("Safe-stop demonstrations:",token_demo["status"],timeout_demo["status"])

    resume_run_id="S1-crash-resume"
    try:
        run_agent(
            s1.message,scenario_id="S1-crash-resume",run_id=resume_run_id,crash_after=2
        )
    except RuntimeError as error:
        if str(error)!="Simulated crash after checkpoint":
            raise
    else:
        raise RuntimeError("The crash-and-resume demonstration did not simulate a crash.")
    resumed=resume_agent(resume_run_id)
    _append_agent_run(resume_run_id,"S1-crash-resume",resumed)
    checkpoint=json.loads((ROOT/"state"/f"{resume_run_id}.json").read_text(encoding="utf-8"))
    expected_tools=["get_order","search_policy","calculate_refund","issue_refund"]
    if resumed["status"]!="REFUND" or abs(_refund_amount(resumed)-128.25)>0.01:
        raise AssertionError("The resumed S1 agent run did not complete the expected refund.")
    if checkpoint["tools_executed"]!=expected_tools:
        raise AssertionError("The crash/resume run repeated or skipped a tool.")
    print("Crash/resume:",{
        "status":resumed["status"],
        "refund_inr":_refund_amount(resumed),
        "tools_executed":checkpoint["tools_executed"],
    })
    print("\nAll project stages completed. Review data/, outputs/, and state/.")


if __name__=="__main__":
    main()
