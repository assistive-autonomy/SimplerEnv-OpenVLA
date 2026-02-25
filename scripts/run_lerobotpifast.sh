#!/usr/bin/env bash
set -euo pipefail

export XDG_RUNTIME_DIR=/usr/lib
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

model_name=lerobotpifast

tasks=(
    bridge.sh    
#    drawer_variant_agg.sh
#    drawer_visual_matching.sh
#    move_near_variant_agg.sh
#    move_near_visual_matching.sh
#    pick_coke_can_variant_agg.sh
#    pick_coke_can_visual_matching.sh
#    put_in_drawer_variant_agg.sh
#    put_in_drawer_visual_matching.sh
)

ckpts=(
    /home/robot/models/lerobot_pi0fast_base
)

action_ensemble_temp=-0.8

for ckpt_path in "${ckpts[@]}"; do
    logging_dir="results/$(basename "$ckpt_path")${action_ensemble_temp}"
    mkdir -p "$logging_dir"

    for task in "${tasks[@]}"; do
        echo "Running $task on GPU 0"
        GPU_IDX=0 bash "scripts/$task" "$ckpt_path" "$model_name" "$action_ensemble_temp" "$logging_dir"
    done

    # statistics evalution results
    echo "🚀 all tasks DONE! Calculating metrics..."
    python tools/calc_metrics_evaluation_videos.py \
        --log-dir-root $logging_dir \
        >>$logging_dir/total.metrics
done
