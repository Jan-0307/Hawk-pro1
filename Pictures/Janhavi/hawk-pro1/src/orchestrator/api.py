from fastapi import FastAPI
from pydantic import BaseModel
from .playbook import handle_alert
from .audit import log_event

app = FastAPI(title="HawkGrid Orchestrator")

class Telemetry(BaseModel):
    node_id: str
    metrics: dict

class Alert(BaseModel):
    node_id: str
    alert_type: str
    details: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest(t: Telemetry):
    log_event({"type":"telemetry","node":t.node_id,"metrics":t.metrics})
    return {"ingested": True}

@app.post("/alert")
def alert(a: Alert):
    log_event({"type":"alert","node":a.node_id,"alert_type":a.alert_type})
    actions = handle_alert(a.node_id, a.alert_type, a.details)
    return {"actions": actions}

@app.get("/")
def read_root():
    return {"message": "Orchestrator API is running"}