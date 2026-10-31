"""Part C: privacy-first ticket triage, retries, validation and routing."""
from pathlib import Path
import json,re,random,time
import pandas as pd
import mock_llm
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"data/raw"; OUT=ROOT/"outputs"
MAX_INPUT_TOKENS=400; BASE_DELAY=.05
MAX_RETRIES=3
PATTERNS=[("email",re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),"[EMAIL]"),
("phone",re.compile(r"\b[6-9]\d{9}\b"),"[PHONE]"),("order_id",re.compile(r"ORD-\d{6}"),"[ORDER_ID]")]
ZERO="""CLASSIFY each support ticket as one of billing, delivery, technical, or fraud. Return JSON only in the form {\"category\": \"...\", \"reason\": \"...\"}."""
FEW=ZERO+"""
EXAMPLES:
Text: Unauthorized transaction, I suspect fraud. -> {\"category\": \"fraud\", \"reason\": \"unauthorized transaction\"}
Text: My order arrived late. -> {\"category\": \"delivery\", \"reason\": \"late delivery\"}
"""
def redact(text):
    counts={}
    for name,pat,repl in PATTERNS:
        text,n=pat.subn(repl,text); counts[name]=n
    return text,counts
def retry_chat(messages,ticket_id):
    for retry_no in range(MAX_RETRIES+1):
        try: return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_no>=MAX_RETRIES: raise
            wait=BASE_DELAY*(2**retry_no)*random.uniform(.9,1.15)
            print(f"retry ticket={ticket_id} retry={retry_no} wait={wait:.6f}s")
            time.sleep(wait)
def run_pipeline(prompt=FEW,token_budget=100000,output_path=None):
    tickets=pd.read_csv(RAW/"tickets.csv",keep_default_na=False)
    seg_path=OUT/"customer_segments.csv"
    seg=pd.read_csv(seg_path) if seg_path.exists() else pd.DataFrame(columns=["customer_id","segment_name","is_premium"])
    segmap=seg.set_index("customer_id").to_dict("index")
    mock_llm.reset(); random.seed(42); total_tokens=0; rows=[]; redaction_totals={k:0 for k,_,_ in PATTERNS}
    redacted_texts={}
    for _,ticket in tickets.iterrows():
        safe,counts=redact(str(ticket.text))
        redacted_texts[ticket.ticket_id]=safe
        for key,count in counts.items():
            redaction_totals[key]+=count
    for safe in redacted_texts.values():
        assert all(pattern.search(safe) is None for _,pattern,_ in PATTERNS)
    for _,t in tickets.iterrows():
        txt=str(t.text); category=""; status=""; tokens=0
        if not txt.strip(): status="skipped_empty"
        elif mock_llm.est_tokens(txt)>MAX_INPUT_TOKENS: status="flagged_long"
        else:
            safe=redacted_texts[t.ticket_id]
            if total_tokens>=token_budget: status="budget_exhausted"
            else:
                response=retry_chat([{"role":"system","content":prompt},{"role":"user","content":safe}],t.ticket_id)
                assert response is not None
                assert response.usage.total_tokens >= 0
                assert all(pattern.search(safe) is None for _,pattern,_ in PATTERNS)
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
    result=pd.DataFrame(rows)
    if output_path is None:
        output_path=OUT/("triage_results.csv" if token_budget>=100000 and prompt==FEW else "triage_budget_demo.csv")
    result.to_csv(output_path,index=False)
    classified=result.category.ne("")
    accuracy=float((result.loc[classified,"category"]==tickets.loc[classified,"true_category"]).mean()) if classified.any() else 0.0
    confusion=pd.crosstab(
        tickets.loc[classified,"true_category"],result.loc[classified,"category"],
        rownames=["true_category"],colnames=["predicted_category"],dropna=False,
    )
    return result,{
        "tokens":total_tokens,
        "classified_tickets":int(classified.sum()),
        "accuracy":accuracy,
        "tokens_per_classified_ticket":total_tokens/int(classified.sum()) if classified.any() else 0.0,
        "confusion_matrix":confusion,
        "redactions":{**redaction_totals,"total":sum(redaction_totals.values()),"remaining":0},
        "status_counts":result.status.value_counts().to_dict(),
    }
