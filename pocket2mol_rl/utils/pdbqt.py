import subprocess as sp
from pathlib import Path
from typing import Optional, Union

import meeko
import torch
from rdkit import Chem


def parse_sdf(sdf_file: Path):
    """
    Parse sdf file into a list of rdkit mol objects
    Args:
        sdf_file: path to the sdf file
    Returns:
        list of rdkit mol objects
    """
    suppl = Chem.SDMolSupplier(str(sdf_file))
    mols = [mol for mol in suppl if mol is not None]
    return mols


def sdf2pdbqt(
    sdf_file: Union[str, Path],
    pdbqt_file: Optional[Union[str, Path]] = None,
    method="meeko",
) -> Path:
    # obabel {ligand_id}.sdf -O{ligand_id}.pdbqt

    sdf_file = Path(sdf_file)
    if not sdf_file.exists():
        raise FileNotFoundError(f"{sdf_file} not found")
    if pdbqt_file is None:
        pdbqt_file = sdf_file.with_suffix(".pdbqt")
    else:
        pdbqt_file = Path(pdbqt_file)
    if pdbqt_file.exists():
        return pdbqt_file

    if method == "meeko":
        mol = parse_sdf(sdf_file)[0]

        protonated_mol = Chem.AddHs(mol)

        meeko_prep = meeko.MoleculePreparation(hydrate=False)
        meeko_prep.prepare(protonated_mol)

        pdbqt_string = meeko_prep.write_pdbqt_string()
        with open(pdbqt_file, "w") as f:
            f.write(pdbqt_string)
    elif method == "obabel":
        result = sp.run(
            ["obabel", str(sdf_file), "-O", str(pdbqt_file)],
            stderr=sp.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(result.stderr)
    else:
        raise ValueError(f"Unknown sdf2pdbqt method {method}")

    return pdbqt_file


class PDB2PDBQTError(Exception):
    pass


class NoPrepareReceptor4Error(PDB2PDBQTError):
    def __str__(self):
        return "You seem not to have AutoDockTools.Utilities.prepare_receptor.py from https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3 in your environment"


def pdb2pdbqt(
    receptor_pdb_file: Path, receptor_pdbqt_file: Optional[Path] = None
) -> Path:
    assert receptor_pdb_file.suffix == ".pdb"
    if receptor_pdbqt_file is None:
        receptor_pdbqt_file = receptor_pdb_file.with_suffix(".pdbqt")
    assert receptor_pdbqt_file.suffix == ".pdbqt"

    if receptor_pdbqt_file.exists():
        return receptor_pdbqt_file

    result = sp.run(
        [
            "python",
            "-m",
            "AutoDockTools.Utilities24.prepare_receptor4",
            "-r",
            str(receptor_pdb_file),
            "-o",
            str(receptor_pdbqt_file),
            "-A",
            "hydrogens",
            "-U",
            "nphs_lps_waters_deleteAltB",
        ],
        stderr=sp.PIPE,
        text=True,
    )
    if result.returncode != 0:
        if "No module named" in result.stderr:
            raise NoPrepareReceptor4Error()
        raise PDB2PDBQTError(result.stderr)

    return receptor_pdbqt_file


def filter_and_save_pdbqt(
    coord_filter_fn,
    src_pdbqt_file: Path,
    tgt_pdbqt_file: Path,
    update_atom_serial=True,
    tensor_input=False,
):
    src_pdbqt_file = Path(src_pdbqt_file)
    tgt_pdbqt_file = Path(tgt_pdbqt_file)
    if not src_pdbqt_file.exists():
        raise FileNotFoundError(src_pdbqt_file)
    if tgt_pdbqt_file.exists():
        print(f"{tgt_pdbqt_file} already exists, skipping")
        return

    updated_lines = []
    with open(src_pdbqt_file, "r") as src_f:
        new_atom_serial_number = 0

        for line in src_f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            coord = [x, y, z]
            if tensor_input:
                coord = torch.FloatTensor(coord)
            if not coord_filter_fn(coord):
                continue

            new_atom_serial_number += 1

            updated_line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
            if update_atom_serial:
                updated_line = (
                    updated_line[:6]
                    + f"{new_atom_serial_number:5d}"
                    + updated_line[11:]
                )

            updated_lines.append(updated_line)

    tgt_pdbqt_file.parent.mkdir(exist_ok=True, parents=True)
    with open(tgt_pdbqt_file, "w") as tgt_f:
        tgt_f.writelines(updated_lines)
