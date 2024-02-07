import pickle
import random
from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski
from rdkit.Chem.rdchem import Mol as RdkitMol
from tqdm import tqdm

from pocket2mol_rl.evaluation.metric.basic import QEDScore
from pocket2mol_rl.evaluation.metric.posebusters.modules.distance_geometry import (
    check_geometry,
)
from pocket2mol_rl.evaluation.metric.posebusters.modules.flatness import (
    check_flatness,
)
from pocket2mol_rl.evaluation.metric.sascorer import compute_sa_score
from pocket2mol_rl.evaluation.metric.similarity import get_average_pairwise_similarity
from pocket2mol_rl.utils.batch_computation import group_compute_merge
from pocket2mol_rl.utils.p2m_docking import P2MDockingTask
from pocket2mol_rl.utils.silence import silence_rdkit

silence_rdkit()


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp := Crippen.MolLogP(mol) >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def _get_tanimoto_similarity(mol1, mol2):
    try:
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return None

    return sim


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("-od", "--output_dir", type=Path, required=True)
    parser.add_argument("-pf", "--protein_filename", type=str, default="receptor.pdb")
    parser.add_argument("-sd", "--sdf_dirname", type=str, required=True)
    parser.add_argument("-rf", "--reference_filename", type=str, default="ref.sdf")

    parser.add_argument("-p", "--precomputed", type=Path, required=True)
    parser.add_argument("-ss", "--subsample", type=float)
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="Random seed for subsampling"
    )
    parser.add_argument("-nw", "--num_workers", type=int, default=8)

    return parser.parse_args()


def parse_mol(sdf_file: Path):
    if not sdf_file.exists():
        raise FileExistsError(f"File {sdf_file} does not exist")
    return Chem.SDMolSupplier(str(sdf_file))[0]


def get_file_tuples(args):
    file_pairs = []

    for subdir in args.output_dir.glob("*"):
        if not subdir.is_dir():
            continue
        pdb_file = subdir / args.protein_filename
        sdf_files = list((subdir / args.sdf_dirname).glob("*.sdf"))
        ref_file = subdir / args.reference_filename
        file_pairs.extend([(pdb_file, sdf_file, ref_file) for sdf_file in sdf_files])

    assert len(file_pairs) > 0, file_pairs
    file_pairs.sort()

    if args.subsample is not None:
        random.seed(args.seed)
        random.shuffle(file_pairs)
        num = int(len(file_pairs) * args.subsample)
        file_pairs = file_pairs[:num]

    return file_pairs


def get_lengths(
    pdb_files: List[Path], mols: List[RdkitMol], ref_mols: List[RdkitMol]
) -> Dict[str, List[Optional[int]]]:
    return {"length": [None if mol is None else len(mol.GetAtoms()) for mol in mols]}


def get_target_id(
    pdb_files: List[Path], mols: List[RdkitMol], ref_mols: List[RdkitMol]
) -> Dict[str, List[str]]:
    return {"target_id": [pdb_file.parent.name for pdb_file in pdb_files]}


def get_smiles(
    pdb_files: List[Path], mols: List[RdkitMol], ref_mols: List[RdkitMol]
) -> Dict[str, List[str]]:
    smiles_list = []

    for mol in tqdm(mols):
        try:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            smiles = None
        except BaseException as e:
            raise e
        smiles_list.append(smiles)
    return {"smiles": smiles_list}


def get_diversity(
    pdb_files: List[Path], mols: List[RdkitMol], ref_mols: List[RdkitMol], num_workers=8
):
    def _group_fn(mol, ref_mol, pdb_file):
        return pdb_file

    def _compute_fn(mols, ref_mols, pdb_files):
        assert len(mols) == len(ref_mols) == len(pdb_files) > 0
        pdb_file = pdb_files[0]
        assert all(pdb_file == pdb_file2 for pdb_file2 in pdb_files)

        diversity = get_average_pairwise_similarity(mols)
        return [diversity for _ in range(len(mols))]

    diversity = group_compute_merge(_group_fn, _compute_fn, mols, ref_mols, pdb_files)
    return {
        "diversity": diversity,
    }


def get_tansim(
    pdb_files: List[Path], mols: List[RdkitMol], ref_mols: List[RdkitMol], num_workers=8
) -> Dict[str, List[Optional[float]]]:
    with Pool(num_workers) as pool:
        tansim_list = pool.starmap(
            _get_tanimoto_similarity, tqdm(zip(mols, ref_mols), total=len(mols))
        )
    """
    tansim_list = parallel_starmap(
        _get_tanimoto_similarity,
        [(m, r) for m, r in zip(mols, ref_mols)],
        cpus=num_workers,
        use_pbar=True,
        replace_errors_with_none=True,
    )
    """
    return {"tansim": tansim_list}


def get_qed(
    pdb_files: List[Path], mols: List[RdkitMol], ref_mols: List[RdkitMol]
) -> Dict[str, List[Optional[float]]]:
    qed_scorer = QEDScore()

    qed_list = qed_scorer.evaluate_in_batch(mols, None, pbar=True)

    return {"qed": qed_list}


def _get_flatness_vals(mol):
    try:
        results = check_flatness(mol, raise_error=True)["results"]
    except Exception:
        num_rings = None
        num_valid_rings = None
    except BaseException as e:
        raise e
    else:
        num_rings = results["num_systems_checked"]
        num_valid_rings = results["num_systems_passed"]

    return (num_rings, num_valid_rings)


def get_flatness(
    pdb_files: List[Path], mols: List[RdkitMol], ref_mols: List[RdkitMol], num_workers=1
) -> Dict[str, List[Optional[float]]]:
    with Pool(num_workers) as pool:
        vals_list = pool.starmap(_get_flatness_vals, tqdm(zip(mols), total=len(mols)))
    """
    vals_list = parallel_starmap(
        _get_vals,
        [(mol,) for mol in mols],
        cpus=num_workers,
        use_pbar=True,
        replace_errors_with_none=True,
    )
    """

    num_rings_list = [vals[0] for vals in vals_list]
    num_valid_rings_list = [vals[1] for vals in vals_list]

    return {
        "num_rings": num_rings_list,
        "num_valid_rings": num_valid_rings_list,
    }


def _get_local_geometry_vals(mol):
    try:
        results = check_geometry(mol, raise_error=True)["results"]
    except Exception:
        num_bonds = None
        num_valid_bonds = None
        num_angles = None
        num_valid_angles = None
        num_noncov = None
        num_valid_noncov = None
        num_clashes = None
    except BaseException as e:
        raise e
    else:
        num_bonds = results["number_bonds"]
        num_valid_bonds = results["number_valid_bonds"]
        num_angles = results["number_angles"]
        num_valid_angles = results["number_valid_angles"]
        num_noncov = results["number_noncov_pairs"]
        num_valid_noncov = results["number_valid_noncov_pairs"]
        num_clashes = results["number_clashes"]
    return (
        num_bonds,
        num_valid_bonds,
        num_angles,
        num_valid_angles,
        num_noncov,
        num_valid_noncov,
        num_clashes,
    )


def get_local_geometry(
    pdb_files: List[Path], mols: List[RdkitMol], ref_mols: List[RdkitMol], num_workers=1
) -> Dict[str, List[Optional[float]]]:
    with Pool(num_workers) as pool:
        vals_list = pool.starmap(
            _get_local_geometry_vals, tqdm(zip(mols), total=len(mols))
        )
    """
    vals_list = parallel_starmap(
        _get_vals,
        [(mol,) for mol in mols],
        cpus=num_workers,
        use_pbar=True,
        replace_errors_with_none=True,
    )
    """

    num_bonds_list = [vals[0] for vals in vals_list]
    num_valid_bonds_list = [vals[1] for vals in vals_list]
    num_angles_list = [vals[2] for vals in vals_list]
    num_valid_angles_list = [vals[3] for vals in vals_list]
    num_noncov_list = [vals[4] for vals in vals_list]
    num_valid_noncov_list = [vals[5] for vals in vals_list]
    num_clashes_list = [vals[6] for vals in vals_list]

    return {
        "num_bonds": num_bonds_list,
        "num_valid_bonds": num_valid_bonds_list,
        "num_angles": num_angles_list,
        "num_valid_angles": num_valid_angles_list,
        "num_noncov": num_noncov_list,
        "num_valid_noncov": num_valid_noncov_list,
        "num_clashes": num_clashes_list,
    }


def _get_sa_score(mol):
    if mol is None:
        return None
    try:
        return compute_sa_score(mol)
    except:
        return None


def get_sa_score(
    pdb_files: List[Path],
    mols: List[RdkitMol],
    ref_mols: List[RdkitMol],
    num_workers=8,
):
    with Pool(num_workers) as pool:
        sa_scores = pool.starmap(_get_sa_score, tqdm(zip(mols), total=len(mols)))
    """
    sa_scores = parallel_starmap(
        _get_sa_score,
        [(mol,) for mol in mols],
        cpus=num_workers,
        use_pbar=True,
        replace_errors_with_none=True,
    )
    """

    return {"sa": sa_scores}


def _get_lipinski(mol):
    try:
        return obey_lipinski(mol)
    except:
        return None


def get_lipinski(
    pdb_files: List[Path],
    mols: List[RdkitMol],
    ref_mols: List[RdkitMol],
    num_workers=8,
):
    with Pool(num_workers) as pool:
        lipinski_vals = pool.starmap(_get_lipinski, tqdm(zip(mols), total=len(mols)))

    return {"lipinski": lipinski_vals}


def get_docking_score(
    pdb_files: List[Path],
    mols: List[RdkitMol],
    ref_mols: List[RdkitMol],
    num_workers=1,
) -> Dict[str, List[Optional[float]]]:

    docking_score_list = []
    for pdb_file, mol in tqdm(zip(pdb_files, mols), total=len(mols)):
        if mol is None:
            docking_score_list.append(None)
            continue
        with P2MDockingTask(pdb_file, mol) as task:
            try:
                score = task.run_sync()
            except Exception as e:
                score = None
            docking_score_list.append(score)

    return {
        "docking_score": docking_score_list,
    }


def get_docking_score_better_than_ref(
    docking_scores: List[Optional[float]],
    pdb_files: List[RdkitMol],
    ref_mols: List[RdkitMol],
    ref_files: List[Path],
    ref_precomputed_file: Path,
    num_workers=1,
):
    def _get_shorthand(ref_file: Path):
        names = str(ref_file).split("/")
        assert "test_outputs" in names
        return "/".join(names[names.index("test_outputs") :])

    ref_files = [_get_shorthand(ref_file) for ref_file in ref_files]

    if ref_precomputed_file.exists():
        with open(ref_precomputed_file, "rb") as f:
            ref_docking_score_dict = pickle.load(f)
    else:
        ref_docking_score_dict = {}

    unique_ref_files = [
        ref_file
        for ref_file in set(ref_files)
        if not ref_file in ref_docking_score_dict
    ]
    unique_idxs = [ref_files.index(ref_file) for ref_file in unique_ref_files]
    unique_pdb_files = [pdb_files[idx] for idx in unique_idxs]
    unique_ref_mols = [ref_mols[idx] for idx in unique_idxs]
    unique_ref_docking_scores = get_docking_score(
        unique_pdb_files, unique_ref_mols, None, num_workers=num_workers
    )["docking_score"]

    for file, score in zip(unique_ref_files, unique_ref_docking_scores):
        ref_docking_score_dict[file] = score
    ref_docking_score_dict.update(
        {
            file: score
            for file, score in zip(unique_ref_files, unique_ref_docking_scores)
        }
    )

    ref_precomputed_file.parent.mkdir(parents=True, exist_ok=True)
    with open(ref_precomputed_file, "wb") as f:
        pickle.dump(ref_docking_score_dict, f)

    ref_docking_scores = [ref_docking_score_dict[ref_file] for ref_file in ref_files]
    docking_score_better_than_ref = [
        None if (val1 is None or val2 is None) else val1 < val2
        for val1, val2 in zip(docking_scores, ref_docking_scores)
    ]

    return {
        "ref_docking_score": ref_docking_scores,
        "docking_score_better_than_ref": docking_score_better_than_ref,
    }


def get_metrics(args):
    metrics_file = args.precomputed / "metrics.pkl"
    if metrics_file.exists():
        with open(metrics_file, "rb") as f:
            d = pickle.load(f)
    else:
        file_tuples = get_file_tuples(args)
        pdb_files = [t[0] for t in file_tuples]
        sdf_files = [t[1] for t in file_tuples]
        mols = [parse_mol(file) for file in sdf_files]
        ref_files = [t[2] for t in file_tuples]
        ref_mols = [parse_mol(file) for file in ref_files]

        # prepare pdbqt file if it doesn't exist
        print("Preparing pdbqt files")
        unique_pdb_files = list(set(pdb_files))
        for pdb_file in tqdm(unique_pdb_files):
            pdbqt_file = pdb_file.with_suffix(".pdbqt")
            assert pdbqt_file.exists(), pdbqt_file

        d = {}
        fns = [
            get_diversity,
            get_tansim,
            get_target_id,
            get_lipinski,
            get_sa_score,
            get_lengths,
            get_smiles,
            get_qed,
            get_flatness,
            get_local_geometry,
            get_docking_score,
        ]
        for fn in fns:
            precomputed_file = args.precomputed / f"{fn.__name__}_results.pkl"
            if precomputed_file.exists():
                print(f"Loading {fn.__name__}")
                with open(precomputed_file, "rb") as f:
                    vals = pickle.load(f)
            else:
                if fn in [
                    get_tansim,
                    get_lipinski,
                    get_sa_score,
                    get_docking_score,
                    get_flatness,
                    get_local_geometry,
                ]:
                    kwargs = {"num_workers": args.num_workers}
                else:
                    kwargs = {}
                print(f"Computing {fn.__name__}")
                vals = fn(pdb_files, mols, ref_mols, **kwargs)
                precomputed_file.parent.mkdir(parents=True, exist_ok=True)
                with open(precomputed_file, "wb") as f:
                    pickle.dump(vals, f)
            d.update(vals)

        precomputed_file = (
            args.precomputed / f"get_docking_score_better_than_ref_results.pkl"
        )
        if precomputed_file.exists():
            print(f"Loading get_docking_score_better_than_ref")
            with open(precomputed_file, "rb") as f:
                vals = pickle.load(f)
        else:
            docking_scores = d["docking_score"]
            ref_precomputed_file = args.precomputed.parent / "ref_docking_score.pkl"
            vals = get_docking_score_better_than_ref(
                docking_scores,
                pdb_files,
                ref_mols,
                ref_files,
                ref_precomputed_file,
                num_workers=args.num_workers,
            )
            with open(precomputed_file, "wb") as f:
                pickle.dump(vals, f)
        d.update(vals)

        # Remove None values
        if not all(len(d[k]) == len(mols) for k in d):
            for k in d:
                print(k, len(d[k]))
            raise Exception("here")
        any_none = [any(d[k][i] is None for k in d) for i in range(len(mols))]
        for k in d:
            d[k] = np.array([d[k][i] for i in range(len(mols)) if not any_none[i]])

        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "wb") as f:
            pickle.dump(d, f)

    return d


if __name__ == "__main__":
    args = parse_args()
    d = get_metrics(args)
