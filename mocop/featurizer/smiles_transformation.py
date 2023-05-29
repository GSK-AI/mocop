import functools

import numpy as np
from openeye import oechem, oegraphsim

from featurizer.molgraph import MolGraph


@functools.lru_cache(maxsize=None)
def smiles2fp(
    smiles: str,
    numbits=1024,
    minradius=0,
    maxradiu=2,
    atype=oegraphsim.OEFPAtomType_DefaultCircularAtom,
    btype=oegraphsim.OEFPBondType_DefaultCircularBond,
):
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smiles)
    fp = oegraphsim.OEFingerPrint()
    oegraphsim.OEMakeCircularFP(
        fp,
        mol,
        numbits,
        minradius,
        maxradiu,
        atype,
        btype,
    )
    fp_vec = [fp.IsBitOn(i) for i in range(fp.GetSize())]
    return np.array(fp_vec).astype(float)


@functools.lru_cache(maxsize=None)
def smiles2graph(smiles, adj_type="norm_id_adj_mat", explicit_H_node=None, **kwargs):
    mol = MolGraph(smiles, explicit_H_node)
    adj_mat = getattr(mol, adj_type)
    node_feat = mol.node_feat
    return adj_mat, node_feat


@functools.lru_cache(maxsize=None)
def smiles2inchikey(smiles: str):
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smiles)
    return oechem.OEMolToSTDInChIKey(mol)


@functools.lru_cache(maxsize=None)
def inchi2smiles(inchikey: str):
    mol = oechem.OEGraphMol()
    oechem.OEInChIToMol(mol, inchikey)
    return oechem.OEMolToSmiles(mol)
