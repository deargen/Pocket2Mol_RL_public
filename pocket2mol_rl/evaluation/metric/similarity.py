from rdkit import Chem, DataStructs
from pocket2mol_rl.utils.mol import RdkitMol
from typing import List
from tqdm import tqdm
import numpy as np


def get_average_pairwise_similarity(mols: List[RdkitMol]):
    fps = []
    for mol in mols:
        try:
            fp = Chem.RDKFingerprint(mol)
            fps.append(fp)
        except:
            fp = None
    assert len(fps) > 0
    assert not any([fp is None for fp in fps])
    vals = []
    pbar = tqdm(total=len(fps) * (len(fps) - 1) // 2)
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.FingerprintSimilarity(fps[i], fps[j])
            vals.append(sim)
            pbar.update(1)
    pbar.close()
    return np.mean(vals)
