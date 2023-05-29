# MoCoP: Molecule-Morphology Contrastive Pretraining for Transferable Molecular Representation

Cuong Q. Nguyen, Dante Pertusi, Kim M. Branson

[[`GSK AI`](https://gsk.ai/)] [[`Paper`](https://arxiv.org/abs/2305.09790)] [[`Blog`](https://gsk.ai/blogs/mocop-multi-modal-representation-of-molecular-graphs-and-cellular-morphologies/)] [[`BibTeX`](#citing-mocop)]
![img](https://gsk.ai/media/5u1jw0lk/cuong-blog-figure2_transparent_v2.png?width=781&height=781&mode=max)

---
## Instructions
#### Cloning and setting up your environment
```bash
git clone https://github.com/GSK-AI/mocop.git
cd mocop
conda env create --name mocop --file environment.yaml
source activate mocop
source .env
```

#### Setting OE_LICENSE 
This step requires the OpenEye license file and is necessary for running src/featurize.py. Change `<path>` to the appropriate directory.
```bash
export OE_LICENSE=<path>/oe_license.txt
```

## Quickstart
#### Finetuning pretrained MoCoP on custom datasets with random splits
Prepare data using the following schema and save as CSV.

| | smiles | task_1 | task_2 | ... | task_N |
| :----: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|0| <smiles_1>  | y<sub>1,1</sub> | y<sub>1,2</sub> | ... | y<sub>1,N</sub> |
|1| <smiles_2>  | y<sub>2,1</sub> | y<sub>2,2</sb> | ... | y<sub>2,N</sub> |
|...| ...  | ...  | ... | ... | ... |

As an example, a dataset with 3 tasks would look something like below.
```csv
,smiles,task_1,task2,task_3
0,smiles_1,y_11,y_12,y_13
1,smiles_2,y_21,y_22,y_23
...
```
Run finetuning from a single MoCoP checkpoint stored in `models/` with random splits. Checkpoints and training artifacts are stored at `models/finetune`
```bash
DATA_PATH=<Path to generated CSV file>

CKPT_PATH=models/jump_mocop_seed_0_split_0/version_0/checkpoints/best-ckpt-remapped.ckpt

python bin/train.py -cn finetune.yml \
                        model._args_.0=$CKPT_PATH \
                        dataloaders.dataset.data_path=$DATA_PATH \
                        dataloaders.splits=null \
                        trainer.logger.save_dir=models \
                        trainer.logger.name=finetune
```
#### Finetuning using pre-specified splits
Prepare your train, validation, and test split CSV files using the following schema. In short, each split file contains a single column `index` that corresponds to the row index of your [data table](#finetuning-pretrained-mocop-on-custom-datasets-with-random-splits) and specifies the set of rows in that split.
|   | index |
| :-------------: | :-------------: |
| 0 | 2 |
| 1 | 10 |
| 2 | 32 |
| ... | ... |
in CSV format, a split file would look like
```csv
,index
0,2
1,10
2,32
...,...
```
We can now finetune our models by replacing `dataloaders.splits=null` with specific split files in the config at runtime.
```bash
DATA_PATH=<Path to generated CSV file>
TRAIN_SPLIT_PATH=<PATH to train split CSV file>
VAL_SPLIT_PATH=<PATH to validation split CSV file>
TEST_SPLIT_PATH=<PATH to test split CSV file>

CKPT_PATH=models/jump_mocop_seed_0_split_0/version_0/checkpoints/best-ckpt-remapped.ckpt

python bin/train.py -cn finetune.yml \
                        model._args_.0=$CKPT_PATH \
                        dataloaders.dataset.data_path=$DATA_PATH \
                        dataloaders.splits.train=$TRAIN_SPLIT_PATH \
                        dataloaders.splits.val=$VAL_SPLIT_PATH \
                        dataloaders.splits.test=$TEST_SPLIT_PATH \
                        trainer.logger.save_dir=models \
                        trainer.logger.name=finetune
```

#### Finetuning using all MoCoP checkpoints
All MoCoP checkpoints are stored in `models/` and can be accessedby looping over them.
```bash
DATA_PATH=<Path to generated CSV file>

CKPTS=(
    models/jump_mocop_seed_0_split_0/version_0/checkpoints/best-ckpt-remapped.ckpt
    models/jump_mocop_seed_1_split_1/version_0/checkpoints/best-ckpt-remapped.ckpt
    models/jump_mocop_seed_2_split_2/version_0/checkpoints/best-ckpt-remapped.ckpt
)

for CKPT_PATH in ${CKPTS[@]}; do
    python bin/train.py -cn finetune.yml \
                            model._args_.0=$CKPT_PATH \
                            dataloaders.dataset.data_path=$DATA_PATH \
                            dataloaders.splits=null \
                            trainer.logger.save_dir=models \
                            trainer.logger.name=finetune
done
```
## Reproducing experiments
Set the necessary environment variables
| Variable  | Description |
| ------------- | ------------- |
| `$CONDA_ENV`  | Name of [conda environment](#cloning-and-setting-up-your-environment) |
| `$DATA_DIR`  | Directory with processed JUMP-CP data   |
| `$SAVE_DIR`  | Output directory for model training  |

Download and preprocess ChEMBL20 and JUMP-CP compound data
```bash
source data/download_and_preprocess.sh $DATA_DIR $CONDA_ENV
```
Pretraining on JUMP-CP
```bash
# MoCoP
source exp/train_jump_mocop.sh $SAVE_DIR $DATA_DIR $CONDA_ENV
# Multitask
source exp/train_jump_multitask.sh $SAVE_DIR $DATA_DIR $CONDA_ENV
```
Remapping model `state_dict` for finetuning
```bash
source exp/remap_state_dict_mocop.sh $SAVE_DIR
source exp/remap_state_dict_multitask.sh $SAVE_DIR
```
Finetuning on ChEMBL20
```bash
# Training from scratch
source exp/train_chembl20_fromscratch.sh $SAVE_DIR $CONDA_ENV
# MoCoP finetune
source exp/train_chembl20_mocop.sh $SAVE_DIR $CONDA_ENV
# MoCoP linear probe
source exp/train_chembl20_mocop_linear.sh $SAVE_DIR $CONDA_ENV
# Multitask finetune
source exp/train_chembl20_multitask.sh $SAVE_DIR $CONDA_ENV
```

Finetuning on ChEMBL20 using MoCoP checkpoints in `models/`
```bash
# MoCoP finetune
source exp/train_chembl20_mocop.sh models $CONDA_ENV
# MoCoP linear probe
source exp/train_chembl20_mocop_linear.sh models $CONDA_ENV
```

## License
MoCoP code is released under the [GPLv3 license](LICENSE-GPLv3) and MoCoP weights are released under the [CC-BY-NC-ND 4.0 license](LICENSE-CC-BY-NC-ND-4.0).


## Citing MoCoP
```
@misc{nguyen2023mocop,
	title={Molecule-Morphology Contrastive Pretraining for Transferable Molecular Representation},
	author={Nguyen, Cuong Q. and Pertusi, Dante and Branson, Kim M.},
	journal={arXiv:2305.09790},
	year={2023},
}
```