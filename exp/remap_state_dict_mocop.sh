SAVE_DIR=$1

for SEED in {0..2}; do
    SPLIT=${SEED}
    CKPT_PATH=${SAVE_DIR}/jump_mocop_seed_${SEED}_split_${SPLIT}/version_0/checkpoints/best_ckpt.ckpt
    OUTPUT_PATH=$(dirname $CKPT_PATH)/best-ckpt-remapped.ckpt

    python bin/remap_state_dict.py -i $CKPT_PATH -o $OUTPUT_PATH --map_from "encoder_a" --map_to "model"
    python bin/remap_state_dict.py -i $OUTPUT_PATH -o $OUTPUT_PATH --map_from "model.fc_layers.0" --map_to "model.fc_layers.0.0"
    echo $OUTPUT_PATH
done