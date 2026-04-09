#!/bin/bash
# Start tool server on a dedicated machine for Metis training.
#
# Usage:
#   bash examples/train/start_tool_server.sh [PORT] [WORKERS]
#
# Example:
#   bash examples/train/start_tool_server.sh 30569 32
#
# Then pass the URL to the training script:
#   REMOTE_TOOL_SERVER_URL=http://<server_ip>:<port>/get_observation \
#       bash examples/train/train_metis.sh 0.15

set -e

PORT=${1:-30569}
WORKERS=${2:-32}
HOST="0.0.0.0"

export PYTHONPATH=$(dirname $0)/../..:$PYTHONPATH

SERVER_IP=$(hostname -i | awk '{print $1}')
LOG_FILE="tool_server_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Metis tool server..."
echo "  Host: ${HOST}:${PORT}"
echo "  Workers: ${WORKERS}"
echo "  Log: ${LOG_FILE}"

nohup python -m verl_tool.servers.serve \
    --host ${HOST} \
    --port ${PORT} \
    --tool_type "metis" \
    --workers_per_tool ${WORKERS} \
    --log_level info \
    > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Health check
MAX_WAIT=120
WAITED=0
while [ ${WAITED} -lt ${MAX_WAIT} ]; do
    if ! ps -p ${SERVER_PID} > /dev/null 2>&1; then
        echo "ERROR: Server process died during startup"
        tail -n 50 "${LOG_FILE}"
        exit 1
    fi
    if curl -s -f -m 3 "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo "Tool server ready!"
        echo "  URL: http://${SERVER_IP}:${PORT}/get_observation"
        echo ""
        echo "To use in training:"
        echo "  REMOTE_TOOL_SERVER_URL=http://${SERVER_IP}:${PORT}/get_observation bash examples/train/train_metis.sh 0.15"
        echo ""
        echo "To stop: kill ${SERVER_PID}"
        exit 0
    fi
    [ $((WAITED % 10)) -eq 0 ] && [ ${WAITED} -gt 0 ] && echo "  Waiting... (${WAITED}s / ${MAX_WAIT}s)"
    sleep 2
    WAITED=$((WAITED + 2))
done

echo "ERROR: Server failed to start after ${MAX_WAIT}s"
tail -n 50 "${LOG_FILE}"
kill -9 ${SERVER_PID} 2>/dev/null || true
exit 1
