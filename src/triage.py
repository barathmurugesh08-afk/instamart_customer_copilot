"""Part C: privacy-first ticket triage, retries, validation and routing."""
from pathlib import Path
import json,re,random,time
import pandas as pd
import mock_llm
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"data/raw"; OUT=ROOT/"outputs"
MAX_INPUT_TOKENS=400; BASE_DELAY=.05
PATTERNS=[("email",re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),"[EMAIL]"),
("phone",re.compile(r"\b[6-9]\d{9}\b"),"[PHONE]"),("order_id",re.compile(r"ORD-\d{6}"),"[ORDER_ID]")]
ZERO="""CLASSIFY. Allowed categories: billing, delivery, technical, fraud. Return JSON only in the form {\"category\":\"...\",\"reason\":\"...\"}."""
FEW=ZERO+"""
EXAMPLES:
Text: Unauthorized transaction, I suspect fraud. -> {\"category\":\"fraud\",\"reason\":\"unauthorized transaction\"}
Text: My order arrived late. -> {\"category\":\"delivery\",\"reason\":\"late delivery\"}
Text: App crashes at login. -> {\"category\":\"technical\",\"reason\":\"app crash\"}
"""
def redact(text):
    counts={}
    for name,pat,repl in PATTERNS:
        text,n=pat.subn(repl,text); counts[name]=n
    return text,counts
def retry_chat(messages,ticket_id):
    for retry_no in range(4):
        try: return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_no>=3: raise
            wait=BASE_DELAY*(2**retry_no)*random.uniform(.9,1.15)
            print(f"retry ticket={ticket_id} retry={retry_no} wait={wait:.4f}s")
            time.sleep(wait)
def run_pipeline(prompt=FEW,token_budget=100000):
    tickets=pd.read_csv(RAW/"tickets.csv",keep_default_na=False)
    seg_path=OUT/"customer_segments.csv"
    seg=pd.read_csv(seg_path) if seg_path.exists() else pd.DataFrame(columns=["customer_id","segment_name","is_premium"])
    segmap=seg.set_index("customer_id").to_dict("index")
    mock_llm.reset(); random.seed(42); total_tokens=0; rows=[]; redaction_totals={k:0 for k,_,_ in PATTERNS}
    for _,t in tickets.iterrows():
        txt=str(t.text); category=""; status=""; tokens=0
        if not txt.strip(): status="skipped_empty"
        elif mock_llm.est_tokens(txt)>MAX_INPUT_TOKENS: status="flagged_long"
        else:
            safe,counts=redact(txt)
            for k,v in counts.items(): redaction_totals[k]+=v
            assert all(not pat.search(safe) for _,pat,_ in [(n,p,r) for n,p,r in PATTERNS])
            if total_tokens>=token_budget: status="budget_exhausted"
            else:
                response=retry_chat([{"role":"system","content":prompt},{"role":"user","content":safe}],t.ticket_id)
                tokens=response.usage.total_tokens; total_tokens+=tokens
                content=response.choices[0].message.content or ""
                try:
                    obj=json.loads(content); cat=obj.get("category")
                    if cat in {"billing","delivery","technical","fraud"}: category=cat; status="classified_json"
                    else: status="flagged_invalid_output"
                except (json.JSONDecodeError,TypeError):
                    found=[w for w in re.findall(r"[a-z]+",content.lower()) if w in {"billing","delivery","technical","fraud"}]
                    if len(set(found))==1: category=found[0]; status="classified_keyword"
                    else: status="flagged_invalid_output"
        s=segmap.get(t.customer_id,{})
        if status in {"skipped_empty","flagged_long","flagged_invalid_output","budget_exhausted"}: route,priority="human_review","NORMAL"
        elif category=="fraud": route,priority="security_team","HIGH"
        elif bool(s.get("is_premium",False)): route,priority="premium_desk","HIGH"
        else: route,priority="support_queue","NORMAL"
        rows.append({"ticket_id":t.ticket_id,"customer_id":t.customer_id,"status":status,"category":category,
          "segment_name":s.get("segment_name",""),"priority":priority,"route":route,"tokens_used":tokens})
    OUT.mkdir(exist_ok=True)
    result=pd.DataFrame(rows); result.to_csv(OUT/("triage_results.csv" if token_budget>=100000 and prompt==FEW else "triage_budget_demo.csv"),index=False)
    return result,{"tokens":total_tokens,"redactions":redaction_totals,"status_counts":result.status.value_counts().to_dict()}
