OUTPUT_DIR=$1
CONDA_ENV=$2


echo "Uncompressing ChEMBL20 data and splits"
tar -xzvf data/chembl20.tar.gz --directory data/

echo "Uncompressing JUMP-CP splits"
tar -xzvf data/jump.tar.gz --directory data/

echo "Cloning JUMP-CP metadata repo"
git clone https://github.com/jump-cellpainting/datasets
METADATA_PATH=datasets

echo "Downloading and normalizing JUMP-CP compound plates"
sbatch  --time=5-00 \
        --mem=40G \
        --array=0-1729 \
        --cpus-per-task=4 \
        --partition=cpu \
        --wait \
        --export=CONDA_ENV=${CONDA_ENV},OUTPUT_DIR=${OUTPUT_DIR},METADATA_PATH=${METADATA_PATH} \
        --wrap "module load miniconda && \
                source activate \${CONDA_ENV} && \
                source ./.env && \
                python data/_jump_download_single_plate.py -o \${OUTPUT_DIR} -m \${METADATA_PATH}"

echo "Aggregating and cleaning JUMP-CP"
python data/_jump_aggregate.py -d $OUTPUT_DIR -o $OUTPUT_DIR --is_centered

echo "Done!"