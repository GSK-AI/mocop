#!/bin/bash

SAVE_DIR=$1
CONDA_ENV=$2

for FRAC in {1,5,10,25,50,100}; do
    for SEED in {0..2}; do
        for SPLIT in {1..3}; do
            sbatch  --time=5-00 \
                    --output=${SAVE_DIR}/slurm-chembl20-fromscratch-%j.out \
                    --mem=120G \
                    --cpus-per-task=12 \
                    --gres=gpu:a6000:1 \
                    --partition=preempted-gpu \
                    --export=SPLIT=${SPLIT},SEED=${SEED},SAVE_DIR=${SAVE_DIR},FRAC=${FRAC},CONDA_ENV=${CONDA_ENV} \
                    --wrap "module load anaconda3 && \
                            source activate \${CONDA_ENV} && \
                            source ./.env && \
                            python bin/train.py -cn train_ggnn_chembl20.yml \
                                                    seed=\${SEED} \
                                                    dataloaders.splits.train=data/chembl20/chembl20-frac\${FRAC}-split\${SPLIT}-train.csv \
                                                    dataloaders.splits.val=data/chembl20/chembl20-split\${SPLIT}-val.csv \
                                                    dataloaders.splits.test=data/chembl20/chembl20-split\${SPLIT}-test.csv \
                                                    dataloaders.num_workers=12 \
                                                    trainer.logger.save_dir=\${SAVE_DIR} \
                                                    trainer.logger.name=chembl20_fromscratch_frac_\${FRAC}_split_\${SPLIT}_seed_\${SEED}"
        done
    done
done
