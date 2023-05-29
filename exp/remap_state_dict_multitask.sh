SAVE_DIR=$1

for SEED in {0..2}; do
    SPLIT=${SEED}
    CKPT_PATH=${SAVE_DIR}/jump_multitask_seed_${SEED}_split_${SPLIT}/version_0/checkpoints/best_ckpt.ckpt
    OUTPUT_PATH=$(dirname $CKPT_PATH)/best-ckpt-remapped.ckpt

    python bin/remap_state_dict.py -i $CKPT_PATH -o $OUTPUT_PATH --map_from "model.fc_layers.1" --map_to "none"
    echo $OUTPUT_PATH