#!/bin/bash

SAVE_DIR=$1
DATA_DIR=$2
CONDA_ENV=$3

# Training
for SEED in {0..2}; do
    SPLIT=${SEED}
    sbatch  --time=5-00 \
            --output=${SAVE_DIR}/slurm-jump-mocop-%j.out \
            --mem=500G \
            --cpus-per-task=48 \
            --gres=gpu:a6000:1 \
            --partition=preempted-gpu \
            --export=SEED=${SEED},SAVE_DIR=${SAVE_DIR},SPLIT=${SPLIT},DATA_DIR=${DATA_DIR},CONDA_ENV=${CONDA_ENV} \
            --wrap "module load anaconda3 && \
                    source activate \${CONDA_ENV} && \
                    echo \${PWD} && \
                    which python && \
                    source ./.env && \
                    python bin/train.py -cn jump_mocop.yml \
                                            seed=\${SEED} \
                                            dataloaders.dataset.data_path=\${DATA_DIR}/centered.filtered.parquet \
                                            dataloaders.splits.train=data/jump/jump-compound-split-\${SPLIT}-train.csv \
                                            dataloaders.splits.val=data/jump/jump-compound-split-\${SPLIT}-val.csv \
                                            dataloaders.splits.test=data/jump/jump-compound-split-\${SPLIT}-test.csv \
                                            trainer.logger.save_dir=\${SAVE_DIR} \
                                            trainer.logger.name=jump_mocop_seed_\${SEED}_split_\${SPLIT}"
done
