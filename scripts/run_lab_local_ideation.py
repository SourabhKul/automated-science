#!/usr/bin/env python3
"""One bounded, hash-pinned local ideation batch; never executes model text."""
import hashlib, json, os, resource, signal, sys, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
ROOT=Path(__file__).resolve().parents[1]
RUN="restart-local-ideation-20261102-v1"
OUT=ROOT/"artifacts/research_lab"/RUN
PROTO=ROOT/"research_lab/protocols/restart-local-ideation-2026-11-02.md"
REQDIR=ROOT/"research_lab/requests/restart-2026-11-02"
BASE="http://127.0.0.1:8080/v1"
MODEL="qwen3.8-flash-next-iq3_s"
PINS={"protocol":"1032837c2fd79917290e27865b5252ee0f0cc26048cda287ddb7af9eed7821e8",
"ecology":"732a802841892933ed05503e783eb2c4acd22d7f0f3593e957a8472d119b5de4",
"hydrology":"effc508d4d4e5733ba24e418fea766d15b81ec7ee92e949fcf731f28fb0690f5"}
ORDER=("ecology","hydrology")
MAX_BODY,MAX_PROMPT,MAX_BYTES=2*1024**2,20000,8*1024**2
MAX_RSS,MAX_TIME,MAX_RECEIPT=512*1024**2,300,128*1024
REQUIRED={"hypothesis_alternatives","smallest_decisive_test","nuisance_and_identifiability_risks","baseline_and_reference","leakage_and_units","reasons_to_reject","next_stage_recommendation"}
class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None
OPENER=build_opener(ProxyHandler({}),NoRedirect())
class Stop(Exception): pass
def sha(b): return hashlib.sha256(b).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def size(): return sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
def rss():
    n=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform=="darwin" else n*1024)
def sample(r):
    n=rss(); r["rss_samples_bytes"].append({"at_utc":utc(),"peak_bytes":n})
    if n>MAX_RSS: raise Stop("sampled client RSS exceeded 512 MiB")
def put(name,data,r=None,replace=False):
    p,t=OUT/name,OUT/(name+".tmp")
    if (p.exists() and not replace) or t.exists(): raise Stop("refusing artifact overwrite: "+name)
    if len(data)>(MAX_RECEIPT if name=="receipt.json" else MAX_BODY): raise Stop("file cap: "+name)
    peak=size()+len(data)+(0 if name=="receipt.json" else MAX_RECEIPT)
    if peak>MAX_BYTES: raise Stop("8 MiB artifact transaction cap")
    if r: r["transaction_peak_bytes"]=max(r.get("transaction_peak_bytes",0),peak)
    with t.open("xb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(t,p); return size()
def save(r):
    r["updated_at_utc"]=utc()
    for _ in range(2):
        old=(OUT/"receipt.json").stat().st_size if (OUT/"receipt.json").exists() else 0
        b=json.dumps(r,sort_keys=True,indent=2).encode()
        r["retained_bytes_after_commit"]=size()-old+len(b)
        r["transaction_peak_bytes"]=max(r.get("transaction_peak_bytes",0),size()+len(b))
    put("receipt.json",json.dumps(r,sort_keys=True,indent=2).encode(),r,True)
def alarm(_sig,_frame): raise TimeoutError("request timed out")
def fetch(path,data,deadline):
    left=deadline-time.monotonic()
    if left<=0: return None,b"",0.0,"300-second budget exhausted"
    timeout=min(120.0,left); q=Request(BASE+path,data=data,method="POST" if data is not None else "GET")
    q.add_header("Accept","application/json")
    if data is not None: q.add_header("Content-Type","application/json")
    start=time.monotonic(); signal.setitimer(signal.ITIMER_REAL,timeout)
    try:
        try: resp=OPENER.open(q,timeout=timeout)
        except HTTPError as e: resp=e
        status=getattr(resp,"status",getattr(resp,"code",None)); final_url=resp.geturl(); b=bytearray(); err=None
        try:
            length=resp.headers.get("Content-Length")
            if length is not None and int(length)>MAX_BODY: err="Content-Length exceeds 2 MiB"
            else:
                while True:
                    c=resp.read(min(65536,MAX_BODY-len(b)+1))
                    if not c: break
                    if len(c)>MAX_BODY-len(b): err="response body exceeds 2 MiB"; break
                    b.extend(c)
        except Exception as e: err=type(e).__name__+": "+str(e)
        finally: resp.close()
        if final_url!=BASE+path: err="response URL did not match fixed localhost endpoint"
        return status,bytes(b),time.monotonic()-start,err,final_url
    except Exception as e: return None,b"",time.monotonic()-start,type(e).__name__+": "+str(e),None
    finally: signal.setitimer(signal.ITIMER_REAL,0)
def load_req(name):
    b=(REQDIR/(name+".json")).read_bytes()
    if sha(b)!=PINS[name]: raise Stop(name+" hash pin mismatch")
    o=json.loads(b); n=sum(len(m["content"].encode("utf-8")) for m in o["messages"])
    if (o.get("model")!=MODEL or o.get("temperature")!=0.2 or o.get("max_tokens")!=1500 or
        o.get("stream") is not False or n>MAX_PROMPT or len(b)>MAX_PROMPT+4096): raise Stop(name+" request contract failed")
    return b,o,n
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); deadline=start+MAX_TIME
    r={"run_id":RUN,"status":"preflight","started_at_utc":utc(),"hash_pins":PINS,
       "protocol_sha256":None,"request_sha256":{},"client_sha256":None,"preflight":{},
       "chat_attempts":[],"failed_chat_attempts":0,"rss_measurement":"client process only; inference server excluded",
       "rss_samples_bytes":[],"transaction_peak_bytes":0,"failure":None}
    save(r)
    try:
        pb=PROTO.read_bytes()
        if sha(pb)!=PINS["protocol"]: raise Stop("protocol hash pin mismatch")
        reqs={n:load_req(n) for n in ORDER}; r["protocol_sha256"]=sha(pb)
        r["request_sha256"]={n:sha(reqs[n][0]) for n in ORDER}; r["client_sha256"]=sha(Path(__file__).read_bytes())
        r["pre_bindings"]={"protocol":r["protocol_sha256"],"requests":r["request_sha256"],"client":r["client_sha256"]}
        for n in ORDER: put(n+".request.json",reqs[n][0],r)
        save(r); sample(r); signal.signal(signal.SIGALRM,alarm)
        began=utc(); status,body,elapsed,err,final_url=fetch("/models",None,deadline); put("models.response.json",body,r)
        r["preflight"]={"status":status,"response_sha256":sha(body),"response_bytes":len(body),
                        "started_at_utc":began,"finished_at_utc":utc(),"elapsed_seconds":elapsed,
                        "final_url":final_url,"error":err}
        save(r)
        if status!=200 or err: raise Stop("model-list preflight failed")
        models=json.loads(body); ids=[x.get("id") for x in models.get("data",[]) if isinstance(x,dict)]
        r["preflight"].update({"returned_model_ids":ids,"exact_model_present":MODEL in ids}); save(r)
        if MODEL not in ids: raise Stop("exact model absent; no POST made")
        for name in ORDER:
            raw,obj,pbytes=reqs[name]; sample(r)
            a={"request":name,"requested_model":MODEL,"request_sha256":sha(raw),"request_bytes":len(raw),
               "prompt_bytes":pbytes,"started_at_utc":utc()}; r["chat_attempts"].append(a); save(r)
            status,body,elapsed,err,final_url=fetch("/chat/completions",raw,deadline); put(name+".response.json",body,r)
            post={"status":status,"response_sha256":sha(body),"response_bytes":len(body),"elapsed_seconds":elapsed,
                  "final_url":final_url,"error":err,"usage":"unknown","returned_model":None,"finish_reason":None}; a["post"]=post
            if status!=200 or err:
                r["failed_chat_attempts"]+=1; save(r); raise Stop(name+" POST failed")
            try:
                result=json.loads(body); choice=result["choices"][0]; msg=choice["message"]
                reason=choice.get("finish_reason")
                post.update({"returned_model":result.get("model"),"finish_reason":reason,
                             "usage":result.get("usage") if result.get("usage") is not None else "unknown"})
                valid=result.get("model")==MODEL and isinstance(msg.get("content"),str) and reason=="stop"
                if valid:
                    try: answer=json.loads(msg["content"]); valid=isinstance(answer,dict) and REQUIRED.issubset(answer)
                    except Exception: valid=False
                post["answer_schema_valid"]=valid; post["refusal_present"]=bool(msg.get("refusal"))
            except Exception as e:
                r["failed_chat_attempts"]+=1; save(r); raise Stop("response envelope failure: "+str(e))
            post["response_complete"]=bool(valid and not post["refusal_present"])
            post["finished_at_utc"]=utc(); sample(r); save(r)
            if not post["response_complete"]:
                r["failed_chat_attempts"]+=1; r["status"]="stopped_on_refusal_or_malformed_answer"
                raise Stop(name+" answer retained; no retry or repair")
        if time.monotonic()>deadline: raise Stop("300-second total processing budget exhausted")
        post={"protocol":sha(PROTO.read_bytes()),"requests":{n:sha((REQDIR/(n+".json")).read_bytes()) for n in ORDER},
              "client":sha(Path(__file__).read_bytes())}
        r["post_bindings"]=post; save(r)
        if post!=r["pre_bindings"] or post["protocol"]!=PINS["protocol"] or post["requests"]!={n:PINS[n] for n in ORDER}:
            raise Stop("pre/post source bindings differ")
        r["status"]="completed_two_calls"
    except Exception as e:
        r["failure"]=type(e).__name__+": "+str(e)
        if r["status"]!="stopped_on_refusal_or_malformed_answer": r["status"]="stopped_failure"
    finally:
        r["total_elapsed_seconds"]=round(time.monotonic()-start,3); r["finished_at_utc"]=utc()
        try: r["final_retained_bytes"]=size(); save(r)
        except Exception as e:
            r["status"]="receipt_persist_failed"; r["failure"]="final receipt publication failed: "+str(e)
            print(r["failure"],file=sys.stderr)
    print(json.dumps({"run_id":RUN,"status":r["status"],"output":str(OUT),"client_sha256":r["client_sha256"],"request_sha256":r["request_sha256"]}))
    return 0 if r["status"]=="completed_two_calls" else 1
if __name__=="__main__": sys.exit(main())
