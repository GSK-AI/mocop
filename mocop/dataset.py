from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from featurizer.smiles_transformation import (inchi2smiles, smiles2fp,
                                              smiles2graph)


class SupervisedGraphDataset(Dataset):
    def __init__(
        self, data_path, cmpd_col="smiles", cmpd_col_is_inchikey=False, pad_length=0
    ):
        if "parquet" in data_path:
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)

        self.df = self.df.set_index(cmpd_col)
        if cmpd_col_is_inchikey:
            self.df.index = [inchi2smiles(s) for s in self.df.index]
        self.df = self.df[[c for c in self.df.columns if not c.startswith("Metadata")]]
        self.unique_smiles = self.df.index
        self.pad_length = pad_length

    def __len__(self):
        return len(self.unique_smiles)

    def _pad(self, adj_mat, node_feat, atom_vec):
        p = self.pad_length - len(atom_vec)
        if p >= 0:
            adj_mat = F.pad(adj_mat, (0, p, 0, p), "constant", 0)
            node_feat = F.pad(node_feat, (0, 0, 0, p), "constant", 0)
            atom_vec = F.pad(atom_vec, (0, 0, 0, p), "constant", 0)
        return adj_mat, node_feat, atom_vec

    def __getitem__(self, index):
        smiles = self.unique_smiles[index]
        adj_mat, node_feat = smiles2graph(smiles)
        adj_mat = torch.FloatTensor(adj_mat)
        node_feat = torch.FloatTensor(node_feat)
        atom_vec = torch.ones(len(node_feat), 1)
        cmpd_feat = self._pad(adj_mat, node_feat, atom_vec)

        labels = self.df.loc[smiles]

        if len(labels.shape) > 1 and len(labels) > 1:
            labels = labels.sample(1).iloc[0]

        labels = torch.FloatTensor(labels.values)
        return {
            "inputs": {"x_a": [torch.FloatTensor(f) for f in cmpd_feat]},
            "labels": labels,
        }


class SupervisedGraphDatasetJUMP(SupervisedGraphDataset):
    def __init__(self, *args, **kwargs):
        super(SupervisedGraphDataset, self).__init__(*args, **kwargs)
        self.unique_smiles = self.df.index.unique()


class DualInputDatasetJUMP(Dataset):
    def __init__(self, data_path):
        if "parquet" in data_path:
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)

        self.smiles_col = "Metadata_SMILES"
        if self.smiles_col not in self.df.columns:
            self.df[self.smiles_col] = [
                inchi2smiles(s) if s is not None else None
                for s in self.df["Metadata_InChI"]
            ]

        self.unique_smiles = [
            s for s in self.df[self.smiles_col].unique() if s is not None
        ]

        self.morph_col = [c for c in self.df.columns if not c.startswith("Metadata_")]
        self.smiles2mask = {}

    def _create_index(self):
        smiles = self.df[self.smiles_col].values
        return {s: np.argwhere(smiles == s).reshape(-1) for s in smiles}

    def __len__(self):
        return len(self.unique_smiles)

    def __getitem__(self, index):
        smiles = self.unique_smiles[index]
        cmpd_feat = smiles2fp(smiles)

        df = self.df[self.df[self.smiles_col] == smiles]
        morph_feat = df.sample(1)[self.morph_col].values.astype(float).flatten()

        labels = torch.Tensor([-1])
        return {
            "inputs": {
                "x_a": torch.FloatTensor(cmpd_feat),
                "x_b": torch.FloatTensor(morph_feat),
            },
            "labels": labels,
        }


class DualInputGraphDatasetJUMP(DualInputDatasetJUMP):
    def __init__(self, pad_length, *args, **kwargs):
        super(DualInputGraphDatasetJUMP, self).__init__(*args, **kwargs)
        self.pad_length = pad_length

    def _pad(self, adj_mat, node_feat, atom_vec):
        p = self.pad_length - len(atom_vec)
        if p >= 0:
            adj_mat = F.pad(adj_mat, (0, p, 0, p), "constant", 0)
            node_feat = F.pad(node_feat, (0, 0, 0, p), "constant", 0)
            atom_vec = F.pad(atom_vec, (0, 0, 0, p), "constant", 0)
        return adj_mat, node_feat, atom_vec

    def __getitem__(self, index):
        smiles = self.unique_smiles[index]
        adj_mat, node_feat = smiles2graph(smiles)
        adj_mat = torch.FloatTensor(adj_mat)
        node_feat = torch.FloatTensor(node_feat)
        atom_vec = torch.ones(len(node_feat), 1)
        cmpd_feat = self._pad(adj_mat, node_feat, atom_vec)

        try:
            mask = self.smiles2mask[smiles]
        except KeyError:
            mask = self.df[self.smiles_col] == smiles
            self.smiles2mask[smiles] = mask
        df = self.df[mask]
        morph_feat = df.sample(1)[self.morph_col].values.astype(float).flatten()
        labels = torch.Tensor([-1])
        return {
            "inputs": {
                "x_a": [torch.FloatTensor(f) for f in cmpd_feat],
                "x_b": torch.FloatTensor(morph_feat),
            },
            "labels": labels,
        }


def _split_data(dataset: Dataset, splits: Dict[str, str]) -> Dict[str, Dataset]:
    if splits is None:
        unique_smiles = dataset.unique_smiles
        total_smiles = len(unique_smiles)
        train_idx = np.random.choice(
            total_smiles, size=int(0.9 * total_smiles), replace=False
        )
        val_idx = [i for i in range(total_smiles) if i not in train_idx]
        return {
            "train": Subset(dataset, train_idx),
            "val": Subset(dataset, val_idx),
            "test": Subset(dataset, val_idx),
        }

    assert "train" in splits and "val" in splits
    split_dataset = {}
    for k, v in splits.items():
        print(f"Split {k}: {v}")
        df_split = pd.read_csv(v)
        if "index" in df_split.columns:
            idx = df_split["index"].values
        else:
            split_smiles = df_split["SMILES"].unique()
            idx = [
                i
                for i, smiles in enumerate(dataset.unique_smiles)
                if smiles in split_smiles
            ]
        split_dataset[k] = Subset(dataset, idx)
    return split_dataset
