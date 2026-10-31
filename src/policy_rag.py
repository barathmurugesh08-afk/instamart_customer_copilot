"""Part D (retrieval): chunk policy files, TF-IDF search and grounded answers."""
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
ROOT=Path(__file__).resolve().parents[1]; POL=ROOT/"data/policies"
class PolicyRAG:
    def __init__(self,chunk_size=40,overlap=10):
        self.chunk_size=chunk_size; self.overlap=overlap; self.chunks=[]
        for path in sorted(POL.glob("*.txt")):
            words=path.read_text(encoding="utf-8").split()
            step=max(1,chunk_size-overlap)
            for start in range(0,len(words),step):
                chunk=words[start:start+chunk_size]
                self.chunks.append({"source":path.name,"text":" ".join(chunk)})
                if start+chunk_size>=len(words): break
        self.vectorizer=TfidfVectorizer()
        self.matrix=self.vectorizer.fit_transform([x["text"] for x in self.chunks])
    def retrieve(self,query,k=3):
        q=self.vectorizer.transform([query]); scores=cosine_similarity(q,self.matrix)[0]
        idx=scores.argsort()[::-1][:k]
        return [{"source":self.chunks[i]["source"],"score":float(scores[i]),"text":self.chunks[i]["text"]} for i in idx]
    def answer(self,question):
        hits=self.retrieve(question,3)
        context="\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        response=mock_llm.chat([{"role":"system","content":"POLICY ASSISTANT: answer only from supplied context. If not present, say I could not find that information in the policy documents."},
          {"role":"user","content":f"CONTEXT:\n{context}\nQUESTION:\n{question}"}])
        return {"answer":response.choices[0].message.content,"sources":list(dict.fromkeys(h["source"] for h in hits)),"retrieved":hits}
def evaluate(rag):
    df=pd.read_csv(ROOT/"data/raw/policy_eval.csv",keep_default_na=False); out=[]
    for _,r in df.iterrows():
        ans=rag.answer(r.question); expected_doc=r.expected_doc; fact=r.expected_fact
        out.append({"question":r.question,"expected_doc":expected_doc,"retrieved_doc":expected_doc in ans["sources"] if expected_doc!="NONE" else False,
          "fact_found":fact in ans["answer"] if fact else False,"fallback":ans["answer"].startswith("I could not find")})
    return pd.DataFrame(out)
