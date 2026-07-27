#!/bin/bash
# =============================================================================
# SGLang Benchmark Suite for Qwen3.6-35B-A3B on 2x RTX 3060 12GB
# Runs short/medium/long generation tests and reports chars/s + tok/s
# =============================================================================

set -euo pipefail

HOST="${HOST:-localhost}"
PORT="${PORT:-3008}"
MODEL="QuantTrio/Qwen3.6-35B-A3B-AWQ"
URL="http://${HOST}:${PORT}/v1/chat/completions"

echo "============================================"
echo "  SGLang Benchmark Suite"
echo "  Server: ${URL}"
echo "  Model:  ${MODEL}"
echo "  Time:   $(date)"
echo "============================================"
echo ""

# Check server is up
if ! curl -s "${URL%/*}/v1/models" 2>/dev/null | grep -q "model"; then
  echo "ERROR: Server not running at $URL"
  echo "Start with: bash sglang-qwen-server-35b.sh --fast"
  exit 1
fi

# Prompts
PROMPTS=(
  "SHORT|Write a short story about a cat who finds a mysterious key.|50"
  "MEDIUM|Explain quantum computing in detail. Cover superposition, entanglement, quantum gates, and practical applications in cryptography and drug discovery.|200"
  "LONG|Write a comprehensive essay about the history of artificial intelligence, from the earliest computer science concepts through modern deep learning. Cover key milestones, important researchers, and the evolution of the field over the decades.|500"
)

# Warmup
echo "Warming up..."
for _ in 1 2; do
  curl -s "$URL" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    >/dev/null 2>&1
done
sleep 2

echo ""
printf "%-10s  %6s  %6s  %8s  %10s  %10s\n" "TEST" "Chars" "Tok" "Time(ms)" "Ch/s(e2e)" "Tok/s(e2e)"
printf "%-10s  %6s  %6s  %8s  %10s  %10s\n" "----------" "------" "------" "--------" "----------" "----------"

for entry in "${PROMPTS[@]}"; do
  IFS='|' read -r label prompt max_tok <<< "$entry"

  START=$(date +%s%N)
  RESP=$(curl -s "$URL" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
      \"max_tokens\": ${max_tok},
      \"chat_template_kwargs\": {\"enable_thinking\": false}
    }" 2>/dev/null)
  END=$(date +%s%N)
  WALL_MS=$(( (END - START) / 1000000 ))

  python3 -c "
import json, sys
d = json.loads('''${RESP}''')
content = d.get('choices',[{}])[0].get('message',{}).get('content','')
usage = d.get('usage',{})
tok = usage.get('completion_tokens', 0)
chars = len(content)
wall = ${WALL_MS}
ch_s = chars * 1000 / wall if wall > 0 else 0
tok_s = tok * 1000 / wall if wall > 0 else 0
print(f'${label:10s}  {chars:6d}  {tok:6d}  {wall:8d}  {ch_s:10.1f}  {tok_s:10.1f}')
" 2>/dev/null

  sleep 1
done

echo ""
echo "--- Decode throughput from server logs ---"
echo "  (check server log for 'gen throughput' lines)"
echo ""

# Show server stats if accessible
echo "--- GPU Memory ---"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo ""
echo "============================================"
