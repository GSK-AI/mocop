#!/bin/bash


SAVE_DIR=$1
CONDA_ENV=$2

declare -a CKPTS=(
        $SAVE_DIR/jump_multitask_seed_0_split_0/version_0/checkpoints/best-ckpt-remapped.ckpt
        $SAVE_DIR/jump_multitask_seed_1_split_1/version_0/checkpoints/best-ckpt-remapped.ckpt
        $SAVE_DIR/jump_multitask_seed_2_split_2/version_0/checkpoints/best-ckpt-remapped.ckpt
)

for FRAC in {1,5,10,25}; do
        for SPLIT in {1..1}; do
                for SEED in "${!CKPTS[@]}"; do
                        RENAMED_BEST_CKPT_PATH=${CKPTS[SEED]}
                        echo $RENAMED_BEST_CKPT_PATH
                        echo $SPLIT
                        sbatch  --time=5-00 \
                                --output=${SAVE_DIR}/slurm-chembl20-multitask-%j.out \
                                --mem=40G \
                                --cpus-per-task=4 \
                                --gres=gpu:a6000:1 \
                                --partition=preempted-gpu \
                                --export=SEED=${SEED},SPLIT=${SPLIT},BEST_CKPT_PATH=${RENAMED_BEST_CKPT_PATH},SAVE_DIR=${SAVE_DIR},FRAC=${FRAC},CONDA_ENV=${CONDA_ENV} \
                                --wrap "module load anaconda3 && \
                                        source activate \${CONDA_ENV} && \
                                        echo \${PWD} && \
                                        which python && \
                                        source ./.env && \
                                        export HYDRA_FULL_ERROR=1 && \
                                        python bin/train.py -cn chembl20_multitask.yml \
                                                                model._args_.0=\${BEST_CKPT_PATH} \
                                                                dataloaders.splits.train=data/chembl20/chembl20-frac\${FRAC}-split\${SPLIT}-train.csv \
                                                                dataloaders.splits.val=data/chembl20/chembl20-split\${SPLIT}-val.csv \
                                                                dataloaders.splits.test=data/chembl20/chembl20-split\${SPLIT}-test.csv \
                                                                dataloaders.num_workers=0 \
                                                                trainer.logger.save_dir=\${SAVE_DIR} \
                                                                trainer.logger.name=chembl20_multitask_frac_\${FRAC}_split_\${SPLIT}_seed_\${SEED}_lr_0.0002_fc_dims_small"
                done
        done
done
