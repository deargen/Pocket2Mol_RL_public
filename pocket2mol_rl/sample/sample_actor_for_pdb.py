import os
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import Optional

import torch
from rdkit import Chem
from tqdm import tqdm

from pocket2mol_rl.rl.model.actor import MolgenActor
from pocket2mol_rl.sample.procedures import generate_exact_number
from pocket2mol_rl.utils.data import (
    pdb_to_pocket_data_from_center,
    pdb_to_pocket_data_from_ref_span,
)
from pocket2mol_rl.utils.mol import get_center_of_sdf_file
from pocket2mol_rl.utils.pdb import get_center_of_pdb_file
from pocket2mol_rl.utils.silence import silence_rdkit

silence_rdkit()


def get_initial_data(
    model_config,
    pdb_path,
    center_from_ref=False,
    ref_path=None,
    bbox_size=None,
    use_ref_span=False,
    distance_to_ref_atom: Optional[float] = None,
):
    if not use_ref_span:
        assert bbox_size is not None
        if center_from_ref:
            assert ref_path is not None
            center = get_center_of_sdf_file(ref_path)
        else:
            center = get_center_of_pdb_file(pdb_path)
            if not isinstance(center, list):
                center = center.tolist()
        data = pdb_to_pocket_data_from_center(pdb_path, center, bbox_size)
    else:
        assert distance_to_ref_atom is not None
        assert ref_path is not None
        data = pdb_to_pocket_data_from_ref_span(
            pdb_path, ref_path, distance_to_ref_atom
        )

    return data


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("-c", "--ckpt_path", type=Path, required=True)
    parser.add_argument("-f", "--file_pattern", type=str, required=True)

    parser.add_argument("-s", "--sdf_subdir", type=str, required=True)
    parser.add_argument("-sm", "--smiles_filename", type=str)
    parser.add_argument("-n", "--num_samples", type=int, default=8)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-fr", "--fr", type=int, default=None)
    parser.add_argument("-to", "--to", type=int, default=None)
    parser.add_argument("-se", "--skip_existing", action="store_true")

    parser.add_argument("-det", "--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-tol", "--batch_tolerance", type=int, default=8)

    parser.add_argument("-gml", "--generation_max_length", type=int, default=60)

    # pdb cropping options
    parser.add_argument("-bb", "--bbox_size", type=float, default=50)
    parser.add_argument("-cfr", "--center_from_ref", action="store_true", help="")
    parser.add_argument(
        "-rs",
        "--use_ref_span",
        action="store_true",
        help="protein atoms within a distance threshold are used for cropping",
    )
    parser.add_argument(
        "-dra",
        "--distance_to_ref_atom",
        type=float,
        default=8.0,
        help="the distance threshold when --use_ref_span is used",
    )

    parser.add_argument(
        "-us",
        "--unique_smiles",
        action="store_true",
    )

    return parser.parse_args()


def get_pdb_sdf_lists(args):
    pdb_files = glob(args.file_pattern)
    if len(pdb_files) == 0:
        raise ValueError(f"No files found with pattern {args.file_pattern}")
    pdb_files = [Path(file).absolute() for file in pdb_files]
    pdb_files.sort()

    if args.fr is not None or args.to is not None:
        assert args.fr is not None and args.to is not None
        assert 0 <= args.fr < args.to <= len(pdb_files)
        pdb_files = pdb_files[args.fr : args.to]

    assert len(set([file.parent for file in pdb_files])) == len(
        pdb_files
    ), "All pdb files must be in different directories."
    sdf_dirs = [file.parent / args.sdf_subdir for file in pdb_files]

    return pdb_files, sdf_dirs


if __name__ == "__main__":
    args = parse_args()

    pdb_files, sdf_dirs = get_pdb_sdf_lists(args)

    print("Loading actor..")
    actor = MolgenActor.from_checkpoint(
        args.ckpt_path,
        device=args.device,
        generation_max_length=args.generation_max_length,
    )
    actor.model.eval()

    for i in tqdm(range(len(pdb_files))):
        if args.deterministic:
            # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.use_deterministic_algorithms(True)
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
        else:
            generator = None

        pdb_file = pdb_files[i]
        sdf_dir = sdf_dirs[i]

        if args.smiles_filename is not None:
            smiles_file = sdf_dir.parent / args.smiles_filename
            if args.skip_existing and smiles_file.exists():
                continue

        print("Preparing data object..")
        ref_file = pdb_file.parent / "ref.sdf"
        initial_data = get_initial_data(
            actor.model.config,
            pdb_file,
            center_from_ref=args.center_from_ref,
            ref_path=ref_file,
            bbox_size=args.bbox_size,
            use_ref_span=args.use_ref_span,
            distance_to_ref_atom=args.distance_to_ref_atom,
        )

        list(
            generate_exact_number(
                actor,
                initial_data,
                args.num_samples,
                sdf_dir=sdf_dir,
                unique_smiles=args.unique_smiles,
                generator=generator,
                max_batch_size=args.batch_size,
                batch_tolerance=args.batch_tolerance,
            )
        )

        if args.smiles_filename is not None:
            sdf_files = list(sdf_dir.glob("*.sdf"))
            sdf_files.sort()
            with open(smiles_file, "w") as f:
                for sdf_file in sdf_files:
                    mol = Chem.SDMolSupplier(str(sdf_file))[0]
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                    f.write(f"{smiles}\n")
