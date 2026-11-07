"""Put a benchmark reference placement back into the running project.

    python tests/bench/restore.py tests/bench/office_truth.json      # or any saved recovery file

Sets every scan's position from the file (matched by label) through the
server, then locks the project.  The manual links stay in the permanent
manual store; this only restores where the scans are.
"""
import json, sys, urllib.request

U = "http://127.0.0.1:8765"


def call(path, body=None):
    req = urllib.request.Request(U + path, None if body is None else json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req))


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    poses = data.get("poses") or data.get("truth", {}).get("poses")
    st = call("/api/state")
    call("/api/lock", {"locked": False})
    n = 0
    for s in st["scans"]:
        if s["label"] in poses:
            call("/api/transform", {"id": s["id"], "matrix": poses[s["label"]]})
            n += 1
    call("/api/lock", {"locked": True})
    print(f"restored {n} of {len(st['scans'])} scans and locked the project")
