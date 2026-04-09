#!/bin/bash
# Multi-node HDPO training for Metis (Qwen3-VL-8B backbone)
#
# Usage:
#   Single Node:
#     bash examples/train/train_metis.sh [W_TOOL]
#
#   Multi-Node (Head):
#     NODE_RANK=0 NNODES=<N> bash examples/train/train_metis.sh [W_TOOL]
#
#   Multi-Node (Worker):
#     NODE_RANK=<1..N-1> MASTER_ADDR=<head_ip> NNODES=<N> bash examples/train/train_metis.sh [W_TOOL]

set -ex

# ==================== User Configuration ====================
W_TOOL=${1:-0.15}                    # HDPO tool-efficiency loss weight
MODEL_PATH=${MODEL_PATH:-"path/to/your/sft-checkpoint"}
TRAIN_DATA=${TRAIN_DATA:-"[data/train.parquet]"}
VAL_DATA=${VAL_DATA:-"[data/val.parquet]"}
WANDB_API_KEY=${WANDB_API_KEY:-""}
REMOTE_TOOL_SERVER_URL=${REMOTE_TOOL_SERVER_URL:-""}

# --- Judge Model (required) ---
# An OpenAI-compatible LLM judge evaluates answer correctness during RL.
# Deploy one with vLLM, e.g.:
#   vllm serve \  
#       --model Qwen/Qwen3-235B-A22B-Instruct-2507 --port 8000 --tensor-parallel-size 8
JUDGE_API_KEY=${JUDGE_API_KEY:-"EMPTY"}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-"http://localhost:8000/v1"}

# --- Search API (required for text_search tool) ---
# The text search tool needs a web search backend. Choose one:
#   Option A: Serper (https://serper.dev)
#   Option B: BrightData SERP API (https://brightdata.com)
#   Option C: SerpApi (https://serpapi.com)
SEARCH_PROVIDER=${SEARCH_PROVIDER:-"serper"}       # "serper" | "serpapi" | "brightdata"
SERPER_API_KEY=${SERPER_API_KEY:-""}                # for serper / serpapi
BRIGHTDATA_API_TOKEN=${BRIGHTDATA_API_TOKEN:-""}    # for brightdata
BRIGHTDATA_ZONE=${BRIGHTDATA_ZONE:-""}              # for brightdata

# --- Tool Sessions ---
METIS_SESSION_DIR=${METIS_SESSION_DIR:-"/tmp/metis_sessions"}
# =============================================================

cd $(dirname $0)/../..

num_nodes=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-6379}

if [ "$NODE_RANK" -eq 0 ] && [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=$(hostname -I | awk '{print $1}')
fi

if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: MASTER_ADDR must be set for worker nodes"
    exit 1
fi

# ==================== Environment Variables ====================
export CODE_DIR=$(pwd)/verl
export PYTHONPATH=$CODE_DIR:$(pwd):$PYTHONPATH

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Judge model
export JUDGE_API_KEY
export JUDGE_BASE_URL

# Search API
export SEARCH_PROVIDER
export SERPER_API_KEY
export BRIGHTDATA_API_TOKEN
export BRIGHTDATA_ZONE

# Tool sessions
export METIS_SESSION_DIR

# Ray memory settings
export RAY_memory_usage_threshold=0.95
export RAY_memory_monitor_refresh_ms=500
export RAY_OBJECT_STORE_MEMORY=400000000000

# NCCL settings
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}

# ==================== Preflight Checks ====================
if [ "$MODEL_PATH" = "path/to/your/sft-checkpoint" ]; then
    echo "ERROR: MODEL_PATH is not set. Please provide the path to your SFT checkpoint."
    exit 1
fi

echo "======== Metis HDPO Training Configuration ========"
echo "  Model:          $MODEL_PATH"
echo "  w_tool:         $W_TOOL"
echo "  Judge URL:      $JUDGE_BASE_URL"
echo "  Search:         $SEARCH_PROVIDER"
echo "  Tool Server:    ${REMOTE_TOOL_SERVER_URL:-auto-start}"
echo "  Nodes:          ${NNODES:-1}"
echo "==================================================="

# ==================== Ray Setup ====================
ray stop --force 2>/dev/null || true
sleep 3

if [ "$NODE_RANK" -eq 0 ]; then
    OBJECT_STORE_MEM=${RAY_OBJECT_STORE_MEMORY:-$(free -b | awk 'NR==2{printf "%.0f", $2 * 0.4}')}
    ray start --head --port=${MASTER_PORT} --num-gpus=8 \
        --object-store-memory=$OBJECT_STORE_MEM --disable-usage-stats
    sleep 15

    # Wait for all nodes to join
    for ((i=1; i<=60; i++)); do
        CONNECTED=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
        echo "Connected nodes: $CONNECTED / $num_nodes"
        [ "$CONNECTED" -ge "$num_nodes" ] && break
        sleep 10
    done
    ray status

    # ==================== Tool Server Setup ====================
    if [ -n "${REMOTE_TOOL_SERVER_URL}" ]; then
        echo "[INFO] Using remote tool server: ${REMOTE_TOOL_SERVER_URL}"
        TOOL_SERVER_URL="${REMOTE_TOOL_SERVER_URL}"
    else
        echo "[INFO] Starting local tool server..."
        HOST=$(hostname -i | awk '{print $1}')
        PORT=$(shuf -i 30000-31000 -n 1)
        TOOL_SERVER_URL="http://${HOST}:${PORT}/get_observation"

        python -m verl_tool.servers.serve \
            --host $HOST \
            --port $PORT \
            --tool_type "metis" \
            --workers_per_tool 16 \
            --log_level info \
            > tool_server.log 2>&1 &
        TOOL_SERVER_PID=$!

        # Health check
        for ((i=1; i<=60; i++)); do
            curl -s -f -m 3 "http://${HOST}:${PORT}/health" > /dev/null 2>&1 && break
            sleep 2
        done
        echo "[INFO] Tool server ready at ${TOOL_SERVER_URL}"
    fi

    # ==================== Training Configuration ====================
    n_gpus_per_node=8
    n=${n:-16}                                           # rollout samples per prompt
    batch_size=$((16 * num_nodes * n_gpus_per_node))
    ppo_mini_batch_size=$((8 * num_nodes * n_gpus_per_node))

    max_prompt_length=24576
    max_response_length=16384
    max_action_length=16384
    max_obs_length=8192
    max_num_batched_tokens=131072

    actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
    infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))

    model_name=$(basename "$MODEL_PATH")
    exp_name="hdpo_${model_name}_wtool${W_TOOL}"
    LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    mkdir -p checkpoints/$exp_name
    mkdir -p rollout_data/$exp_name
    mkdir -p logs

    # Create action stop tokens file
    ACTION_STOP_TOKENS_FILE=$(mktemp /tmp/action_stop_tokens.XXXXXX)
    echo -n '</tool_call>' > "$ACTION_STOP_TOKENS_FILE"
    trap "rm -f $ACTION_STOP_TOKENS_FILE; ray stop --force 2>/dev/null; kill $TOOL_SERVER_PID 2>/dev/null" EXIT

    # ==================== Launch Training ====================
    python3 -m verl_tool.trainer.main_ppo \
        algorithm.adv_estimator=hdpo \
        algorithm.kl_ctrl.kl_coef=0 \
        \
        data.train_files=$TRAIN_DATA \
        data.val_files=$VAL_DATA \
        data.train_batch_size=$batch_size \
        data.dataloader_num_workers=8 \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        data.filter_overlong_prompts=False \
        data.filter_overlong_prompts_workers=8 \
        data.image_patch_size=16 \
        data.truncation='error' \
        \
        reward_model.reward_manager=metis \
        reward_model.launch_reward_fn_async=True \
        \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.trust_remote_code=True \
        ++actor_rollout_ref.nccl_timeout=3600 \
        \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
        actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
        actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_ppo_max_token_len \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.w_acc=1.0 \
        actor_rollout_ref.actor.w_tool=$W_TOOL \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        \
        actor_rollout_ref.agent.tool_call_timeout=400 \
        actor_rollout_ref.agent.enable_agent=True \
        actor_rollout_ref.agent.tool_server_url=$TOOL_SERVER_URL \
        actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
        actor_rollout_ref.agent.max_response_length=$max_response_length \
        actor_rollout_ref.agent.max_start_length=$max_prompt_length \
        actor_rollout_ref.agent.max_obs_length=$max_obs_length \
        actor_rollout_ref.agent.max_action_length=$max_action_length \
        actor_rollout_ref.agent.max_turns=10 \
        actor_rollout_ref.agent.additional_eos_token_ids="[151645]" \
        actor_rollout_ref.agent.mask_observations=True \
        actor_rollout_ref.agent.action_stop_tokens=$ACTION_STOP_TOKENS_FILE \
        actor_rollout_ref.agent.enable_mtrl=True \
        actor_rollout_ref.agent.max_concurrent_trajectories=256 \
        \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.agent.num_workers=32 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.n=$n \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$infer_ppo_max_token_len \
        actor_rollout_ref.rollout.max_num_seqs=2048 \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
        actor_rollout_ref.rollout.enable_prefix_caching=True \
        \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$infer_ppo_max_token_len \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        \
        critic.optim.lr=1e-5 \
        critic.strategy=fsdp2 \
        critic.model.path=$MODEL_PATH \
        critic.model.fsdp_config.fsdp_size=-1 \
        critic.ppo_micro_batch_size_per_gpu=4 \
        critic.ulysses_sequence_parallel_size=1 \
        \
        trainer.logger='[console,wandb]' \
        trainer.project_name=hdpo \
        trainer.experiment_name=$exp_name \
        trainer.val_before_train=False \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=checkpoints/$exp_name \
        trainer.n_gpus_per_node=$n_gpus_per_node \
        trainer.nnodes=$num_nodes \
        trainer.rollout_data_dir=rollout_data/$exp_name \
        trainer.save_freq=50 \
        trainer.test_freq=10000 \
        trainer.total_epochs=10 \
        trainer.total_training_steps=200 \
        2>&1 | tee logs/${exp_name}_${LOG_TIMESTAMP}.log

else
    # Worker node
    sleep 30
    OBJECT_STORE_MEM=${RAY_OBJECT_STORE_MEMORY:-$(free -b | awk 'NR==2{printf "%.0f", $2 * 0.4}')}
    for ((i=1; i<=10; i++)); do
        ray start --address="${MASTER_ADDR}:${MASTER_PORT}" --num-gpus=8 \
            --object-store-memory=$OBJECT_STORE_MEM --disable-usage-stats && break
        sleep 30
    done
    sleep infinity
fi
