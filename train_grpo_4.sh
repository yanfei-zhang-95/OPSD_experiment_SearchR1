export CUDA_VISIBLE_DEVICES=4,5,6,7
# export DATA_DIR='data/nq_search'
export DATA_DIR='data/r1searcher_stage1'

# export DATA_DIR='data/r1searcher_stage1_with_golden_rollout'

WAND_PROJECT='Search-R1'

# export BASE_MODEL='/data/huggingface_models/Qwen2.5-3B'
# export EXPERIMENT_NAME=r1-searcher-r1-grpo-qwen2.5-3b-em
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=r1-searcher-r1-grpo-qwen2.5-7b-it-em

export BASE_MODEL='/data/huggingface_models/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo'
export EXPERIMENT_NAME=r1-searcher-r1-grpo-qwen2.5-3b-em-ContinualNoRLSD

# export BASE_MODEL='/data/huggingface_models/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo'
# export EXPERIMENT_NAME=r1-searcher-r1-grpo-qwen2.5-3b-em-ContinualNoRLSD

export RESUME_CKPT=${RESUME_CKPT:-}
export AUTO_RESUME=${AUTO_RESUME:-1}
WANDB_RUN_ID=${WANDB_RUN_ID:-}
WANDB_RESUME_MODE=${WANDB_RESUME_MODE:-must}

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])
# Default OPSD / RLSD-style setting:
#   +algorithm.opsd_student_scoring_mode=causal_prefix
#   +algorithm.opsd_teacher_mode=stale_ref_policy
#   +algorithm.opsd_teacher_refresh_interval=10
# Resume example:
#   RESUME_CKPT=verl_checkpoints/r1-searcher-r1-grpo-qwen2.5-7b-it-em/actor/global_step_20 bash train_grpo.sh
# Wandb resume example:
#   WANDB_RUN_ID=<old_run_id> RESUME_CKPT=verl_checkpoints/.../actor/global_step_20 bash train_grpo.sh
# Automatic resume:
#   bash train_grpo.sh
#   This auto-detects `wandb/latest-run` and the latest `global_step_*` under `verl_checkpoints/$EXPERIMENT_NAME/actor`.
# Optional ablation:
#   +algorithm.opsd_student_scoring_mode=rollout_old_log_prob
#   +algorithm.opsd_teacher_mode=live_actor
# Optional hindsight peer-trajectory augmentation:
#   Only used when OPSD_HINDSIGHT_INFO_MODE=peer_traj
#   OPSD_HINDSIGHT_INCLUDE_FIRST_CORRECT_PEER_TRAJECTORY=false bash train_grpo_4.sh
# Optional hindsight info mode:
#   peer_traj: only for the original peer-trajectory path
#   golden_rollout: applies to all cases that carry extra_info.golden_rollout
#   OPSD_HINDSIGHT_INFO_MODE=golden_rollout bash train_grpo_4.sh
# Optional OPSD step advantage normalization:
#   none: keep the current token-level OPSD shaping only
#   equal_step_mean_abs: rescale each step after OPSD so step-wise mean(abs(advantage)) matches
#   OPSD_STEP_ADVANTAGE_NORM=equal_step_mean_abs bash train_grpo_4.sh
# Optional OOM fallback:
#   +actor_rollout_ref.rollout.opsd_logprob_chunk_size=16
export OPSD_HINDSIGHT_INCLUDE_FIRST_CORRECT_PEER_TRAJECTORY="${OPSD_HINDSIGHT_INCLUDE_FIRST_CORRECT_PEER_TRAJECTORY:-false}"
export OPSD_HINDSIGHT_INFO_MODE="${OPSD_HINDSIGHT_INFO_MODE:-none}"
export OPSD_STEP_ADVANTAGE_NORM="${OPSD_STEP_ADVANTAGE_NORM:-none}"

EXTRA_ARGS=()
ACTOR_CKPT_DIR="verl_checkpoints/$EXPERIMENT_NAME/actor"
AUTO_RESUME_READY=0
if [ -z "$RESUME_CKPT" ] && [ "$AUTO_RESUME" = "1" ] && [ -d "$ACTOR_CKPT_DIR" ]; then
    LATEST_CKPT=$(find "$ACTOR_CKPT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)
    if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT/trainer_state.pt" ]; then
        RESUME_CKPT="$LATEST_CKPT"
        AUTO_RESUME_READY=1
        echo "[AUTO RESUME] Using checkpoint: $RESUME_CKPT"
    fi
fi

if [ -z "$WANDB_RUN_ID" ] && [ "$AUTO_RESUME" = "1" ] && [ "$AUTO_RESUME_READY" = "1" ] && [ -L "wandb/latest-run" ]; then
    LATEST_WANDB_RUN=$(basename "$(readlink -f wandb/latest-run)")
    WANDB_RUN_ID="${LATEST_WANDB_RUN##*-}"
    echo "[AUTO RESUME] Using wandb run id: $WANDB_RUN_ID"
fi

if [ "$AUTO_RESUME" = "1" ] && [ "$AUTO_RESUME_READY" != "1" ]; then
    RESUME_CKPT=""
    WANDB_RUN_ID=""
    unset WANDB_RUN_ID
    unset WANDB_RESUME_MODE
    echo "[AUTO RESUME] No valid checkpoint found under $ACTOR_CKPT_DIR. Starting a fresh wandb run."
fi

if [ -n "$RESUME_CKPT" ]; then
    EXTRA_ARGS+=("+trainer.resume_from_path=$RESUME_CKPT")
fi
if [ -n "$WANDB_RUN_ID" ]; then
    EXTRA_ARGS+=("trainer.wandb_run_id=$WANDB_RUN_ID")
    EXTRA_ARGS+=("trainer.wandb_resume=$WANDB_RESUME_MODE")
fi



PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=64 \
    data.val_batch_size=125 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.max_start_length=1024 \
    data.max_obs_length=512 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
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
    +algorithm.opsd_student_scoring_mode=causal_prefix \
    +algorithm.opsd_target_span_mode="clean_step_no_observation" \
    +algorithm.opsd_teacher_mode=stale_ref_policy \
    +algorithm.opsd_teacher_refresh_interval=10 \
    +algorithm.opsd_weight_clip=0.2 \
    +algorithm.opsd_mix_lambda_init=0 \
    +algorithm.opsd_mix_lambda_decay_steps=200 \
    +algorithm.opsd_hindsight_include_first_correct_peer_trajectory=${OPSD_HINDSIGHT_INCLUDE_FIRST_CORRECT_PEER_TRAJECTORY} \
    +algorithm.opsd_hindsight_info_mode=${OPSD_HINDSIGHT_INFO_MODE} \
    +algorithm.opsd_step_advantage_norm=${OPSD_STEP_ADVANTAGE_NORM} \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.seed=42 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=20 \
    retriever.url="http://127.0.0.1:8085/retrieve" \
    retriever.topk=3 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee $EXPERIMENT_NAME.log
