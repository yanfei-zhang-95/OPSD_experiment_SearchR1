#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
# 1) Evaluate all default datasets (hotpotqa, musique, 2wikimultihopqa)
# BASE_MODEL=/data/yanfeizhang/OPSD_experiment/Search-R1/verl_checkpoints/r1-searcher-r1-grpo-qwen2.5-3b-it-em-ContinualRLSD/actor/global_step_60 \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# bash scripts/nq_hotpotqa/evaluate.sh
#
# 2) Evaluate a single dataset
# DATA_NAME=hotpotqa \
# BASE_MODEL=/data/yanfeizhang/OPSD_experiment/Search-R1/verl_checkpoints/r1-searcher-r1-grpo-qwen2.5-3b-it-em-ContinualRLSD/actor/global_step_60 \
# bash scripts/nq_hotpotqa/evaluate.sh
#
# 3) Evaluate a custom dataset list
# EVAL_DATASETS="hotpotqa musique 2wikimultihopqa bamboogle" \
# bash scripts/nq_hotpotqa/evaluate.sh

DEFAULT_EVAL_DATASETS="hotpotqa musique 2wikimultihopqa"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-500}"
export RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8085/retrieve}"
export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
export OPSD_STUDENT_SCORING_MODE="${OPSD_STUDENT_SCORING_MODE:-causal_prefix}"
export OPSD_TARGET_SPAN_MODE="${OPSD_TARGET_SPAN_MODE:-clean_step_no_observation}"
export OPSD_TEACHER_MODE="${OPSD_TEACHER_MODE:-stale_ref_policy}"
export OPSD_TEACHER_INCLUDE_FINAL_CORRECTNESS="${OPSD_TEACHER_INCLUDE_FINAL_CORRECTNESS:-False}"
# export BASE_MODEL="${BASE_MODEL:-/data/huggingface_models/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo}"
export BASE_MODEL="${BASE_MODEL:-/data/yanfeizhang/OPSD_experiment/Search-R1/verl_checkpoints/r1-searcher-r1-grpo-qwen2.5-3b-it-em-ContinualNoRLSD/actor/global_step_200}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${EVAL_LOG_DIR:-$SCRIPT_DIR/eval_logs}}"

export VLLM_ATTENTION_BACKEND=XFORMERS

if [ ! -d "$BASE_MODEL" ]; then
    echo "[ERROR] BASE_MODEL path not found: $BASE_MODEL"
    exit 1
fi

sanitize_tag() {
    local raw="$1"
    raw="${raw// /_}"
    raw="${raw//\//_}"
    raw="${raw//:/_}"
    raw="${raw//[^a-zA-Z0-9._-]/_}"
    printf '%s' "$raw"
}

model_tag="$(sanitize_tag "$(basename "$BASE_MODEL")")"
model_parent_tag="$(basename "$(dirname "$BASE_MODEL")")"
if [ "$model_parent_tag" = "actor" ] || [ "$model_parent_tag" = "critic" ] || [ "$model_parent_tag" = "ref" ]; then
    experiment_tag="$(basename "$(dirname "$(dirname "$BASE_MODEL")")")"
else
    experiment_tag="$(basename "$(dirname "$BASE_MODEL")")"
fi
experiment_tag="$(sanitize_tag "${MODEL_CLASS_TAG:-$experiment_tag}")"
if [ -z "$experiment_tag" ] || [ "$experiment_tag" = "." ] || [ "$experiment_tag" = "/" ]; then
    experiment_tag="standalone_model"
fi

export EVAL_LOG_DIR="${EVAL_LOG_ROOT}/${experiment_tag}/${model_tag}"
mkdir -p "$EVAL_LOG_DIR"

summary_file="$EVAL_LOG_DIR/summary.txt"
{
    echo "Evaluation summary"
    echo "base_model=$BASE_MODEL"
    echo "eval_log_root=$EVAL_LOG_ROOT"
    echo "experiment_tag=$experiment_tag"
    echo "model_tag=$model_tag"
    echo "eval_log_dir=$EVAL_LOG_DIR"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo
} > "$summary_file"

resolve_data_dir() {
    case "$1" in
        hotpotqa|rucai_eval_hotpotqa)
            echo "data/rucai_eval_hotpotqa"
            ;;
        musique|rucai_eval_musique)
            echo "data/rucai_eval_musique"
            ;;
        2wikimultihopqa|2wikimultihoptqa|rucai_eval_2wikimultihopqa)
            echo "data/rucai_eval_2wikimultihopqa"
            ;;
        bamboogle|rucai_eval_bamboogle)
            echo "data/rucai_eval_bamboogle"
            ;;
        *)
            echo "data/$1"
            ;;
    esac
}

run_eval() {
    local dataset_name="$1"
    local data_dir="${DATA_DIR:-$(resolve_data_dir "$dataset_name")}"
    local train_files="${TRAIN_FILES:-$data_dir/train.parquet}"
    local val_files="${VAL_FILES:-$data_dir/test.parquet}"
    local log_file="$EVAL_LOG_DIR/${dataset_name}.log"
    local detail_file="$EVAL_LOG_DIR/${dataset_name}.details.jsonl"

    echo "============================================================"
    echo "[EVAL] dataset=$dataset_name"
    echo "[EVAL] data_dir=$data_dir"
    echo "[EVAL] base_model=$BASE_MODEL"
    echo "[EVAL] experiment_tag=$experiment_tag"
    echo "[EVAL] model_tag=$model_tag"
    echo "[EVAL] log_file=$log_file"
    echo "[EVAL] detail_file=$detail_file"
    echo "============================================================"

    if [ ! -f "$train_files" ]; then
        echo "[ERROR] Missing train parquet: $train_files"
        exit 1
    fi
    if [ ! -f "$val_files" ]; then
        echo "[ERROR] Missing val parquet: $val_files"
        exit 1
    fi

    rm -f "$detail_file"

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        data.train_files="$train_files" \
        data.val_files="$val_files" \
        data.train_data_num=null \
        data.val_data_num=null \
        data.train_batch_size=1 \
        data.val_batch_size="$VAL_BATCH_SIZE" \
        data.max_prompt_length=4096 \
        data.max_response_length=1024 \
        data.max_start_length=1024 \
        data.max_obs_length=512 \
        data.return_raw_chat=true \
        data.shuffle_train_dataloader=false \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.path="$BASE_MODEL" \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size=32 \
        actor_rollout_ref.actor.fsdp_config.param_offload=true \
        actor_rollout_ref.actor.fsdp_config.grad_offload=true \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        algorithm.no_think_rl=false \
        +algorithm.opsd_student_scoring_mode="$OPSD_STUDENT_SCORING_MODE" \
        +algorithm.opsd_target_span_mode="$OPSD_TARGET_SPAN_MODE" \
        +algorithm.opsd_teacher_mode="$OPSD_TEACHER_MODE" \
        +algorithm.opsd_teacher_include_final_correctness="$OPSD_TEACHER_INCLUDE_FINAL_CORRECTNESS" \
        actor_rollout_ref.rollout.n_agent=5 \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.actor.state_masking=true \
        trainer.critic_warmup=0 \
        trainer.logger=[] \
        +trainer.val_only=true \
        +trainer.val_before_train=true \
        +trainer.validation_detail_path="$detail_file" \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
        trainer.nnodes=1 \
        max_turns=20 \
        retriever.url="$RETRIEVER_URL" \
        retriever.topk="$RETRIEVER_TOPK" \
        2>&1 | tee "$log_file"

    local metric_line
    metric_line="$(grep -E 'Initial validation metrics|Final validation metrics|val/test_score/' "$log_file" | tail -n 1 || true)"
    {
        echo "dataset=$dataset_name"
        echo "log_file=$log_file"
        echo "detail_file=$detail_file"
        if [ -n "$metric_line" ]; then
            echo "metric=$metric_line"
        else
            echo "metric=[WARN] No validation metric line found"
        fi
        echo
    } >> "$summary_file"
}

if [ -n "${DATA_NAME:-}" ]; then
    run_eval "$DATA_NAME"
else
    for dataset_name in ${EVAL_DATASETS:-$DEFAULT_EVAL_DATASETS}; do
        run_eval "$dataset_name"
    done
fi
