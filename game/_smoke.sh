#!/bin/bash
# End-to-end check for the game web app: start the server, play full games over
# HTTP picking only legal moves, verify static assets + illegal-move rejection.
# Run via the background bash tool (uses sleep). Exits non-zero on any failure.
set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
PY="${PY:-/scratch1/kelidari/envs/coev/bin/python}"
PORT=8011

"$PY" game/server.py --host 127.0.0.1 --port "$PORT" > /tmp/ginserver.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

ok=""
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/api/state" >/dev/null 2>&1; then ok=1; echo "server up after ~${i}s"; break; fi
  kill -0 $SRV 2>/dev/null || { echo "server died:"; tail -30 /tmp/ginserver.log; exit 1; }
  sleep 1
done
[ -n "$ok" ] || { echo "server never came up:"; tail -30 /tmp/ginserver.log; exit 1; }

"$PY" - "$PORT" <<'PYEOF'
import sys, json, urllib.request, urllib.error
BASE=f"http://127.0.0.1:{sys.argv[1]}"
def post(path, body=None):
    data=json.dumps(body or {}).encode()
    req=urllib.request.Request(BASE+path, data=data,
        headers={"Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())

# 1) static assets
for path,needle in [("/", b"<!DOCTYPE"), ("/app.js", b"send"),
                    ("/style.css", b"felt"), ("/deck_images/1_of_spades.png", None)]:
    with urllib.request.urlopen(BASE+path) as r:
        body=r.read()
        assert r.status==200 and body, path
        assert needle is None or needle in body, f"content mismatch {path}"
    print("static OK", path)

def legal_set(v):
    L=v["legal"]; s=set()
    if L["draw_stock"]: s.add(2)
    if L["take_discard"]: s.add(3)
    if L["declare_dead"]: s.add(4)
    if L["gin"]: s.add(5)
    for i in L["discard"]: s.add(6+i)
    for i in L["knock"]: s.add(58+i)
    return s

def pick(v):
    L=v["legal"]
    if L["gin"]: return 5
    if L["knock"]: return 58+L["knock"][0]
    if L["discard"]: return 6+L["discard"][0]
    if L["draw_stock"]: return 2
    if L["take_discard"]: return 3
    if L["declare_dead"]: return 4
    return None

# 2) play full games, only legal moves
results={"win":0,"loss":0,"draw":0}
for g in range(8):
    st,v=post("/api/new_game"); assert st==200 and "error" not in v, v
    steps=0
    while not v["done"] and steps<400:
        assert v["hand_count"] in (10,11), ("bad hand_count", v["hand_count"])
        lv=v.get("opponent_hand_live")
        assert isinstance(lv,list) and len(lv) in (10,11), ("bad opp live", None if lv is None else len(lv))
        # The opponent's hand must be DISJOINT from the player's (one deck) -- this
        # is the regression guard for "opponent shows my hand".
        ph=set(c["idx"] for c in v["hand"]); oh=set(c["idx"] for c in lv)
        assert ph.isdisjoint(oh), ("opponent hand overlaps player hand!", sorted(ph & oh))
        a=pick(v); assert a is not None, ("no legal action", v["legal"])
        st,v=post("/api/action",{"action":a})
        assert st==200 and "error" not in v, ("action error", st, v)
        steps+=1
    assert v["done"], ("game didn't finish", steps)
    assert v.get("opponent_reveal") and len(v["opponent_reveal"]) in (10,11), ("reveal missing", v.get("opponent_reveal"))
    assert isinstance(v.get("opponent_deadwood"), int), ("opp deadwood missing", v.get("opponent_deadwood"))
    rh=set(c["idx"] for c in v["hand"]); ro=set(c["idx"] for c in v["opponent_reveal"])
    assert rh.isdisjoint(ro), ("reveal overlaps player hand!", sorted(rh & ro))
    om=set()
    for m in v.get("opponent_melds",[]): om.update(m)
    assert om <= ro, ("opponent melds not within opponent hand", sorted(om - ro))
    results[v["result"]]+=1
    print(f"game {g}: {v['result']} in {steps} steps, final dw={v['deadwood']}")

# 2b) opponent chooser: both opponents listed, and switching works
with urllib.request.urlopen(BASE+"/api/opponents") as r: opps=json.loads(r.read())
keys=[o["key"] for o in opps]
assert "selfplay" in keys and "pool" in keys and "winrate" in keys and "reward" in keys, keys
print("opponents:", keys)
st,v=post("/api/new_game",{"opponent":"reward"})
assert st==200 and v.get("opponent_key")=="reward", ("switch failed", v.get("opponent_key"))
st,v=post("/api/new_game",{"opponent":"winrate"})
assert st==200 and v.get("opponent_key")=="winrate", ("switch back failed", v.get("opponent_key"))
print("opponent switching OK")

# 3) illegal move is rejected with 400
st,v=post("/api/new_game")
illegal=next(a for a in range(2,110) if a not in legal_set(v))
st2,v2=post("/api/action",{"action":illegal})
print(f"illegal action {illegal}: status={st2} msg={v2.get('error')}")
assert st2==400, "illegal move must be rejected"

print("results:", results)
print("WEB_TEST_PASS")
PYEOF
rc=$?
echo "client exit: $rc"
echo "--- server log tail ---"; tail -6 /tmp/ginserver.log
exit $rc
