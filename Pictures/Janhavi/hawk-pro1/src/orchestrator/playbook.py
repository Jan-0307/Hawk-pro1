import yaml, os
def handle_alert(node_id, alert_type, details):
    pb_file = f"playbooks/{alert_type}.yaml"
    if not os.path.exists(pb_file):
        return []
    pb = yaml.safe_load(open(pb_file))
    actions=[]
    for step in pb.get("steps", []):
        actions.append({"type": step.get("type"), "params": step.get("params")})
    return actions