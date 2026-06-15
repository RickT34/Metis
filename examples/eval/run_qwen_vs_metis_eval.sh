#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# ==================== Required user input ====================
DATA_FILE="${DATA_FILE:-}"

# ==================== Model choices ====================
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
BASE_SERVED_NAME="${BASE_SERVED_NAME:-Qwen3-VL-8B-Instruct}"
METIS_MODEL_PATH="${METIS_MODEL_PATH:-Accio-Lab/Metis-8B-RL}"
METIS_SERVED_NAME="${METIS_SERVED_NAME:-Metis-8B-RL}"

# ==================== 8-GPU vLLM settings ====================
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TP_SIZE="${TP_SIZE:-8}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# ==================== Tool server settings ====================
TOOL_PORT="${TOOL_PORT:-30569}"
TOOL_WORKERS="${TOOL_WORKERS:-16}"
TOOL_SERVER_URL="${TOOL_SERVER_URL:-http://127.0.0.1:${TOOL_PORT}/get_observation}"

# ==================== Evaluation settings ====================
RUN_DIR="${RUN_DIR:-runs/qwen_vs_metis_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-$(dirname "${DATA_FILE:-.}")}"
LIMIT="${LIMIT:-0}"
MAX_TURNS="${MAX_TURNS:-10}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
JUDGE_MODE="${JUDGE_MODE:-exact}"  # exact or llm
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

if [[ -z "${DATA_FILE}" ]]; then
  echo "ERROR: set DATA_FILE=/path/to/eval.jsonl"
  exit 1
fi

mkdir -p "${RUN_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export VLLM_API_KEY JUDGE_API_KEY
export SEARCH_PROVIDER="${SEARCH_PROVIDER:-serper}"
export SERPER_API_KEY="${SERPER_API_KEY:-}"
export BRIGHTDATA_API_TOKEN="${BRIGHTDATA_API_TOKEN:-}"
export BRIGHTDATA_ZONE="${BRIGHTDATA_ZONE:-}"

echo "[config]"
echo "  DATA_FILE=${DATA_FILE}"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  RUN_DIR=${RUN_DIR}"
echo "  BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "  METIS_MODEL_PATH=${METIS_MODEL_PATH}"
echo "  GPU_IDS=${GPU_IDS}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  JUDGE_MODE=${JUDGE_MODE}"
echo "  LIMIT=${LIMIT}"

python examples/eval/validate_eval_data.py --data "${DATA_FILE}" --data-root "${DATA_ROOT}"

VLLM_PID=""
TOOL_PID=""

cleanup() {
  if [[ -n "${TOOL_PID}" ]]; then
    kill "${TOOL_PID}" 2>/dev/null || true
    TOOL_PID=""
  fi
  if [[ -n "${VLLM_PID}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    VLLM_PID=""
  fi
}
trap cleanup EXIT

wait_for_vllm() {
  local log_file="$1"
  echo "[vllm] waiting for ${VLLM_BASE_URL}/models"
  for i in $(seq 1 300); do
    if curl -s -f "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
      echo "[vllm] ready"
      return 0
    fi
    if [[ -n "${VLLM_PID}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "ERROR: vLLM server died"
      tail -n 120 "${log_file}" || true
      exit 1
    fi
    sleep 2
  done
  echo "ERROR: vLLM endpoint did not become ready"
  tail -n 120 "${log_file}" || true
  exit 1
}

start_vllm() {
  local model_path="$1"
  local served_name="$2"
  local log_file="$3"
  cleanup
  echo "[vllm] starting ${served_name} from ${model_path}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  vllm serve "${model_path}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --served-model-name "${served_name}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --trust-remote-code \
    ${VLLM_EXTRA_ARGS} \
    > "${log_file}" 2>&1 &
  VLLM_PID=$!
  wait_for_vllm "${log_file}"
}

start_tool_server() {
  local log_file="$1"
  echo "[tool] starting Metis tool server on port ${TOOL_PORT}"
  python -m verl_tool.servers.serve \
    --host 0.0.0.0 \
    --port "${TOOL_PORT}" \
    --tool_type metis \
    --workers_per_tool "${TOOL_WORKERS}" \
    --log_level info \
    > "${log_file}" 2>&1 &
  TOOL_PID=$!
  for i in $(seq 1 120); do
    if curl -s -f "http://127.0.0.1:${TOOL_PORT}/health" >/dev/null 2>&1; then
      echo "[tool] ready"
      return 0
    fi
    if ! kill -0 "${TOOL_PID}" 2>/dev/null; then
      echo "ERROR: tool server died"
      tail -n 120 "${log_file}" || true
      exit 1
    fi
    sleep 2
  done
  echo "ERROR: tool server did not become ready"
  tail -n 120 "${log_file}" || true
  exit 1
}

judge_args=()
if [[ "${JUDGE_MODE}" == "llm" ]]; then
  if [[ -z "${JUDGE_BASE_URL}" || -z "${JUDGE_MODEL}" ]]; then
    echo "ERROR: JUDGE_MODE=llm requires JUDGE_BASE_URL and JUDGE_MODEL"
    exit 1
  fi
  judge_args=(--judge-mode llm --judge-base-url "${JUDGE_BASE_URL}" --judge-model "${JUDGE_MODEL}")
else
  judge_args=(--judge-mode exact)
fi

echo "[run] original Qwen3-VL no-tool"
start_vllm "${BASE_MODEL_PATH}" "${BASE_SERVED_NAME}" "${RUN_DIR}/base_vllm.log"
python examples/eval/eval_openai_tool_agent.py \
  --data "${DATA_FILE}" \
  --output "${RUN_DIR}/base_no_tool.jsonl" \
  --mode no_tool \
  --model "${BASE_SERVED_NAME}" \
  --model-alias base \
  --base-url "${VLLM_BASE_URL}" \
  --api-key "${VLLM_API_KEY}" \
  --data-root "${DATA_ROOT}" \
  --limit "${LIMIT}" \
  --max-turns "${MAX_TURNS}" \
  --max-tokens-per-turn "${MAX_TOKENS_PER_TURN}" \
  --temperature "${TEMPERATURE}" \
  --resume \
  "${judge_args[@]}"

echo "[run] Metis-8B-RL tool policy"
start_vllm "${METIS_MODEL_PATH}" "${METIS_SERVED_NAME}" "${RUN_DIR}/metis_vllm.log"
start_tool_server "${RUN_DIR}/tool_server.log"
python examples/eval/eval_openai_tool_agent.py \
  --data "${DATA_FILE}" \
  --output "${RUN_DIR}/metis_policy.jsonl" \
  --mode policy \
  --model "${METIS_SERVED_NAME}" \
  --model-alias metis \
  --base-url "${VLLM_BASE_URL}" \
  --api-key "${VLLM_API_KEY}" \
  --tool-url "${TOOL_SERVER_URL}" \
  --data-root "${DATA_ROOT}" \
  --limit "${LIMIT}" \
  --max-turns "${MAX_TURNS}" \
  --max-tokens-per-turn "${MAX_TOKENS_PER_TURN}" \
  --temperature "${TEMPERATURE}" \
  --resume \
  "${judge_args[@]}"

echo "[compare]"
python examples/eval/compare_model_correctness.py \
  --base "${RUN_DIR}/base_no_tool.jsonl" \
  --metis "${RUN_DIR}/metis_policy.jsonl" \
  --base-name "${BASE_SERVED_NAME}" \
  --metis-name "${METIS_SERVED_NAME}" \
  --out-dir "${RUN_DIR}/compare"

echo "[done]"
echo "  base predictions:  ${RUN_DIR}/base_no_tool.jsonl"
echo "  metis predictions: ${RUN_DIR}/metis_policy.jsonl"
echo "  comparison:        ${RUN_DIR}/compare/model_compare_summary.md"
