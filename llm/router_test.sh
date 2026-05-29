#!/bin/bash
# GPU-less integration check for the master/router control plane:
# start a fake worker + the real master on localhost, then verify
# discovery -> health -> routing -> cache (miss then hit) -> Ollama protocol.
#
# Run via the background bash tool (it uses sleep). Exits non-zero on failure.

PY="${PY:-/scratch1/kelidari/envs/coev/bin/python}"
PORT_M=11455
PORT_W=8077
export GINLLM_RUNTIME_DIR="$(mktemp -d)"
cd /home1/kelidari/Adversarial-CoEvolution

echo "runtime=$GINLLM_RUNTIME_DIR  python=$($PY --version 2>&1)"

"$PY" -m llm.fake_worker --port "$PORT_W" --worker-id fake0 --model olmoe-fake \
    > /tmp/fakeworker.log 2>&1 &
FW=$!
"$PY" -m llm.master --port "$PORT_M" --scan-interval 1 \
    > /tmp/routermaster.log 2>&1 &
MA=$!
cleanup() { kill "$FW" "$MA" 2>/dev/null; }
trap cleanup EXIT

# Wait for the master to see the worker as healthy.
HEALTHY=""
for i in $(seq 1 40); do
    H=$(curl -sf "http://127.0.0.1:${PORT_M}/health" 2>/dev/null || true)
    if echo "$H" | grep -q '"n_healthy":1'; then HEALTHY=1; break; fi
    sleep 1
done
echo "health: ${H:-<none>}"
if [ -z "$HEALTHY" ]; then
    echo "FAIL: master never saw a healthy worker"
    echo "--- master log ---"; tail -n 30 /tmp/routermaster.log
    echo "--- worker log ---"; tail -n 30 /tmp/fakeworker.log
    exit 1
fi

gen() {
    curl -sf -XPOST "http://127.0.0.1:${PORT_M}/api/generate" \
        -H 'content-type: application/json' \
        -d "{\"model\":\"x\",\"prompt\":\"$1\",\"options\":{\"num_predict\":32}}"
}

R1=$(gen "deal me a hand"); echo "call1 (expect miss): $R1"
R2=$(gen "deal me a hand"); echo "call2 (expect HIT ): $R2"
R3=$(gen "a different state"); echo "call3 (expect miss): $R3"

echo "stats: $(curl -sf http://127.0.0.1:${PORT_M}/stats)"

# Assertions
fail=0
echo "$R1" | grep -q '"cached":false' || { echo "FAIL: call1 not a miss"; fail=1; }
echo "$R2" | grep -q '"cached":true'  || { echo "FAIL: call2 not a hit";  fail=1; }
echo "$R1" | grep -q 'draw from stock' || { echo "FAIL: call1 missing worker text"; fail=1; }
S=$(curl -sf http://127.0.0.1:${PORT_M}/stats)
echo "$S" | grep -q '"hits":1' || { echo "FAIL: expected 1 cache hit in stats"; fail=1; }

if [ "$fail" = 0 ]; then echo "ROUTER_TEST_PASS"; else echo "ROUTER_TEST_FAIL"; fi
exit $fail
