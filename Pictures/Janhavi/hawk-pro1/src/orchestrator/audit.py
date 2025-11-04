import json, hashlib, os
os.makedirs("logs", exist_ok=True)
def log_event(ev):
    s = json.dumps(ev, sort_keys=True)
    h = hashlib.sha256(s.encode()).hexdigest()
    with open("logs/events.log","a") as f:
        f.write(json.dumps({"hash":h,"event":ev})+"\n")
    return h