"""Part D: database-backed refund tools and a resumable, bounded ReAct loop."""
from pathlib import Path
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import mock_llm
from src.policy_rag import PolicyRAG

ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/"data/instamart.db"
RAW=ROOT/"data/raw"
OUT=ROOT/"outputs"
STATE=ROOT/"state"
rag=PolicyRAG()


def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row=con.execute(
            "select order_id,status,order_amount,delivery_minutes from orders where order_id=?",
            (order_id,),
        ).fetchone()
    return {"error":"order not found"} if not row else dict(zip(
        ["order_id","status","order_amount","delivery_minutes"],row
    ))


def search_policy(query):
    return {"results":rag.retrieve(query,3)}


def calculate_refund(order_amount,delivery_minutes):
    amount=float(order_amount) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0
    return {"refund_inr":round(amount,2)}


def _ledger(tool,order_id,refund_inr):
    OUT.mkdir(exist_ok=True)
    with (OUT/"refund_ledger.jsonl").open("a",encoding="utf-8") as file:
        file.write(json.dumps({
            "timestamp":datetime.now(timezone.utc).isoformat(),
            "tool":tool,
            "order_id":order_id,
            "refund_inr":refund_inr,
        })+"\n")
    return {"refund_inr":refund_inr}


def issue_refund(order_id,refund_inr):
    return _ledger("issue_refund",order_id,refund_inr)


def escalate_to_human(order_id,refund_inr):
    return _ledger("escalate_to_human",order_id,refund_inr)


TOOL_MAP={
    "get_order":get_order,
    "search_policy":search_policy,
    "calculate_refund":calculate_refund,
    "issue_refund":issue_refund,
    "escalate_to_human":escalate_to_human,
}
TOOLS_SPEC=[
    {
        "name":"get_order",
        "description":"Look up an order and return its status, amount, and delivery time.",
        "parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]},
    },
    {
        "name":"search_policy",
        "description":"Retrieve the policy passages most relevant to a query.",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]},
    },
    {
        "name":"calculate_refund",
        "description":"Calculate a full late-delivery refund or zero when the order is not eligible.",
        "parameters":{"type":"object","properties":{
            "order_amount":{"type":"number"},"delivery_minutes":{"type":["number","null"]}
        },"required":["order_amount","delivery_minutes"]},
    },
    {
        "name":"issue_refund",
        "description":"Record an automatically approved refund in the local audit ledger.",
        "parameters":{"type":"object","properties":{
            "order_id":{"type":"string"},"refund_inr":{"type":"number"}
        },"required":["order_id","refund_inr"]},
    },
    {
        "name":"escalate_to_human",
        "description":"Record a refund that requires human approval in the local audit ledger.",
        "parameters":{"type":"object","properties":{
            "order_id":{"type":"string"},"refund_inr":{"type":"number"}
        },"required":["order_id","refund_inr"]},
    },
]


def build_agent_scenarios():
    orders=pd.read_csv(ROOT/"data/clean/orders.csv")
    completed=orders[orders.status.eq("completed")].copy()
    late=completed[completed.delivery_minutes.gt(45)].sort_values("order_id")
    if len(late)<3:
        raise ValueError("The clean order table needs three late completed orders for the agent scenarios.")
    late_small=late[late.order_amount.le(1000)]
    late_large=late[late.order_amount.gt(1000)]
    on_time=completed[completed.delivery_minutes.le(45)].sort_values("order_id")
    cancelled=orders[orders.status.eq("cancelled")].sort_values("order_id")
    if len(late_small)<2 or late_large.empty or on_time.empty or cancelled.empty:
        raise ValueError("The clean order table is missing a required refund-agent scenario.")
    first,second=late_small.iloc[0],late_small.iloc[1]
    high=late_large.iloc[0]
    timely=on_time.iloc[0]
    canceled=cancelled.iloc[0]
    scenarios=[
        ("S1",first.customer_id,f"My order {first.order_id} arrived very late. Please refund it.","REFUND",first.order_amount),
        ("S2",second.customer_id,f"My order {second.order_id} was delivered late. Please refund it.","REFUND",second.order_amount),
        ("S3",high.customer_id,f"My order {high.order_id} arrived very late. Please refund it.","ESCALATE",high.order_amount),
        ("S4",timely.customer_id,f"Please refund my order {timely.order_id}.","DENIED",0),
        ("S5",canceled.customer_id,f"Please refund my cancelled order {canceled.order_id}.","DENIED",0),
        ("S6","C0001","Refund for order ORD-999999 please.","NOT_FOUND",0),
        ("S7",first.customer_id,f"[LOOP] Refund order {first.order_id}.","FALLBACK",0),
    ]
    path=RAW/"agent_scenarios.csv"
    pd.DataFrame(scenarios,columns=[
        "scenario_id","customer_id","message","expected_outcome","expected_refund_inr"
    ]).to_csv(path,index=False)
    return path


def _checkpoint(run_id,state):
    STATE.mkdir(exist_ok=True)
    path=STATE/f"{run_id}.json"
    path.write_text(json.dumps(state,default=str),encoding="utf-8")


def _fallback(status,steps,llm_calls,tokens_used,detail):
    return {
        "status":status,
        "final_answer":f"{status}: hand off to a human agent. {detail}",
        "steps":steps,
        "llm_calls":llm_calls,
        "tokens_used":tokens_used,
    }


def _continue_agent(run_id,state,max_iterations,token_limit,time_limit_seconds,crash_after=None):
    messages=state["messages"]
    tokens=state["tokens_used"]
    steps=state["steps"]
    executed=state["tools_executed"]
    llm_calls=state["llm_calls"]
    start=time.monotonic()
    while llm_calls<max_iterations:
        if time.monotonic()-start>=time_limit_seconds:
            return _fallback("FALLBACK_TIMEOUT",steps,llm_calls,tokens,"Time limit reached.")
        response=mock_llm.chat(messages,tools=TOOLS_SPEC)
        llm_calls+=1
        tokens+=response.usage.total_tokens
        if tokens>token_limit:
            return _fallback("FALLBACK_TOKEN_LIMIT",steps,llm_calls,tokens,"Token limit exceeded.")
        msg=response.choices[0].message
        if not msg.tool_calls:
            content=msg.content or ""
            return {
                "status":content.split(":",1)[0],
                "final_answer":content,
                "steps":steps,
                "llm_calls":llm_calls,
                "tokens_used":tokens,
            }
        tc=msg.tool_calls[0]
        args=tc.arguments
        messages.append({
            "role":"assistant",
            "content":None,
            "tool_calls":[{"id":tc.id,"name":tc.name,"arguments":args}],
        })
        observation=TOOL_MAP[tc.name](**args)
        executed.append(tc.name)
        steps.append({"tool":tc.name,"arguments":args,"observation":observation})
        messages.append({
            "role":"tool",
            "tool_call_id":tc.id,
            "name":tc.name,
            "content":json.dumps(observation),
        })
        state.update({
            "messages":messages,
            "tokens_used":tokens,
            "step_number":len(steps),
            "tools_executed":executed,
            "steps":steps,
            "llm_calls":llm_calls,
        })
        _checkpoint(run_id,state)
        if crash_after is not None and len(executed)==crash_after:
            raise RuntimeError("Simulated crash after checkpoint")
    return _fallback(
        "FALLBACK_MAX_ITERATIONS",steps,llm_calls,tokens,
        "Maximum model-call count reached.",
    )


def run_agent(
    user_message,
    scenario_id="demo",
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after=None,
):
    run_id=run_id or str(uuid.uuid4())
    state={
        "run_id":run_id,
        "scenario_id":scenario_id,
        "messages":[
            {"role":"system","content":"AGENT: Verify orders and follow refund policy using the available tools."},
            {"role":"user","content":user_message},
        ],
        "tokens_used":0,
        "step_number":0,
        "tools_executed":[],
        "steps":[],
        "llm_calls":0,
    }
    return _continue_agent(
        run_id,state,max_iterations,token_limit,time_limit_seconds,crash_after
    )


def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    path=STATE/f"{Path(run_id).name}.json"
    state=json.loads(path.read_text(encoding="utf-8"))
    if state.get("run_id")!=run_id:
        raise ValueError(f"Checkpoint run id does not match: {run_id}")
    return _continue_agent(run_id,state,max_iterations,token_limit,time_limit_seconds)
