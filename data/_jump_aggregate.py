import os
import pandas as pd
import argparse


def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    is_centered = args.is_centered
    dfs = []
    for f in os.listdir(data_dir):
        if is_centered and "centered" not in f:
            continue
        df = pd.read_parquet(os.path.join(data_dir, f))
        dfs.append(df)

    df = pd.concat(dfs)
    df = drop_bad_columns(df)
    df = df.dropna(axis=1)
    filename = f"{'centered.' if is_centered else ''}filtered.parquet"
    filepath = os.path.join(output_dir, filename)
    df.to_parquet(filepath)
    print(f"File saved to {filepath}")


def drop_bad_columns(df):
    cols = [c for c in df.columns if "Metadata" not in c]
    stdev = [df[c].std() for c in cols]

    cols_to_drop = []
    cols_to_drop.extend([cols[i] for i, s in enumerate(stdev) if s < 0.1 or s > 5])
    cols_to_drop.extend([c for c in cols if "Nuclei_Correlation_RWC" in c])
    cols_to_drop.extend([c for c in cols if "Nuclei_Correlation_Manders" in c])
    cols_to_drop.extend([c for c in cols if "Nuclei_Granularity_14" in c])
    cols_to_drop.extend([c for c in cols if "Nuclei_Granularity_15" in c])
    cols_to_drop.extend([c for c in cols if "Nuclei_Granularity_16" in c])

    df = df[[c for c in df.columns if c not in cols_to_drop]]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir")
    parser.add_argument("-o", "--output-dir")
    parser.add_argument("--is_centered", default=False, action="store_true")
    args = parser.parse_args()
    main(args)