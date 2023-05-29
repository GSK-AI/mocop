import argparse
import os

import pandas as pd
from sklearn import preprocessing


def download_single_plate(plate):
    profile_formatter = (
        "s3://cellpainting-gallery/cpg0016-jump/"
        "{Metadata_Source}/workspace/profiles/"
        "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet"
    )

    s3_path = profile_formatter.format(**plate.to_dict())
    df = pd.read_parquet(s3_path, storage_options={"anon": True})
    return df


def main(args):
    metadata_path = args.metadata_path
    output_dir = args.output_dir
    index = args.index
    if index is None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    # Load metadata files from JUMP metadata https://github.com/jump-cellpainting/datasets
    plates = pd.read_csv(os.path.join(metadata_path, "metadata/plate.csv.gz"))
    plates = plates.query('Metadata_PlateType=="COMPOUND"')
    print(len(plates))
    wells = pd.read_csv(os.path.join(metadata_path, "metadata/well.csv.gz"))
    compound = pd.read_csv(os.path.join(metadata_path, "metadata/compound.csv.gz"))
    metadata = compound.merge(wells, on="Metadata_JCP2022")

    plate = plates.iloc[index]
    df = download_single_plate(plate)
    df = metadata.merge(df, on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"])
    print(len(df))

    # Save unprocessed plate features with merged metadata
    name = f"{plate['Metadata_Source']}.{plate['Metadata_Batch']}.{plate['Metadata_Plate']}.{plate['Metadata_Plate']}"
    filename = os.path.join(output_dir, f"{name}.parquet")
    df.to_parquet(filename)
    print(f"Save raw file to {filename}")

    # Preprocessing centering on DMSO
    cols = [c for c in df.columns if not c.startswith("Metadata_")]
    scaler = preprocessing.RobustScaler()
    X = df[df.Metadata_InChI == "InChI=1S/C2H6OS/c1-4(2)3/h1-2H3"][cols].values
    scaler.fit(X)
    df.loc[:, cols] = scaler.transform(df[cols].values)

    # Save preprocessed plate features with merged metadata
    filename = os.path.join(output_dir, f"{name}.centered.parquet")
    df.to_parquet(filename)
    print(f"Save centered file to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metadata-path")
    parser.add_argument("-o", "--output-dir")
    parser.add_argument("-i", "--index", default=None, type=int)
    args = parser.parse_args()
    main(args)
