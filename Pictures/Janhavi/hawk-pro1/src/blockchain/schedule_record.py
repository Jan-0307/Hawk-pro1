import json
import sys
from ledger import compute_root_from_events, push_and_optionally_record

def main():
    root = compute_root_from_events("logs/events.log")
    if not root:
        print(json.dumps({"ok": False, "msg": "no events"}))
        return 1
    res = push_and_optionally_record(root)
    print(json.dumps({"ok": True, "result": res}))
    return 0

if __name__ == "__main__":
    sys.exit(main())