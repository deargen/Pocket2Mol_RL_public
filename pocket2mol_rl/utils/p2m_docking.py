import os
import random
import string
import subprocess
import subprocess as sp
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import numpy as np
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from rdkit.Chem.rdMolAlign import CalcRMS

from pocket2mol_rl.utils.mol import RdkitMol, parse_sdf
from pocket2mol_rl.utils.pdbqt import pdb2pdbqt, sdf2pdbqt
from pocket2mol_rl.utils.qvina import QVINA_PATH
from pocket2mol_rl.utils.reconstruct import reconstruct_from_generated_with_edges


def get_random_id(length=30):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def load_pdb(path):
    with open(path, "r") as f:
        return f.read()


def parse_qvina_outputs(docked_sdf_path, ref_mol):
    suppl = Chem.SDMolSupplier(str(docked_sdf_path))
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp("REMARK").splitlines()[0].split()[2:]
        try:
            rmsd = CalcRMS(ref_mol, mol)
        except:
            rmsd = np.nan
        results.append(
            EasyDict(
                {
                    "rdmol": mol,
                    "mode_id": i,
                    "affinity": float(line[0]),
                    "rmsd_lb": float(line[1]),
                    "rmsd_ub": float(line[2]),
                    "rmsd_ref": rmsd,
                }
            )
        )

    return results


class P2MDockingTask:
    def __init__(self, receptor_pdb_file: Path, mol, use_uff=True, center=None):
        self.entered = False

        self.receptor_pdb_file = Path(receptor_pdb_file)
        if not isinstance(mol, RdkitMol):
            mol = Path(mol)
        self.mol = mol
        self.use_uff = use_uff
        self.provided_center = center

    def __enter__(self):
        self.entered = True

        self._tmp_dir = TemporaryDirectory()
        self.tmp_dir = Path(self._tmp_dir.name)

        if isinstance(self.mol, RdkitMol):
            self.tmp_mol_sdf_file = self.tmp_dir / f"tmp_mol.sdf"
            ligand_rdmol = self.mol
        else:
            self.tmp_mol_sdf_file = self.tmp_dir / self.mol.name
            ligand_rdmol = parse_sdf(self.mol)

        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
        if self.use_uff:
            try:
                not_converge = 10
                while not_converge > 0:
                    flag = UFFOptimizeMolecule(ligand_rdmol)
                    not_converge = min(not_converge - 1, flag * 10)
            except RuntimeError:
                pass

        sdf_writer = Chem.SDWriter(str(self.tmp_mol_sdf_file))
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()

        self.ligand_rdmol = ligand_rdmol
        self.noH_rdmol = Chem.RemoveHs(ligand_rdmol)

        if self.provided_center is None:
            pos = ligand_rdmol.GetConformer(0).GetPositions()
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = self.provided_center

        self.proc = None
        self.results = None
        self.output = None
        self.docked_sdf_file = None

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._tmp_dir.cleanup()

    def run(self, exhaustiveness=16):
        # Prepare receptor (PDB->PDBQT)
        receptor_pdbqt_file = pdb2pdbqt(self.receptor_pdb_file)

        # Prepare ligand
        tmp_mol_pdbqt_file = sdf2pdbqt(self.tmp_mol_sdf_file, method="obabel")

        self.docked_pdbqt_file = (
            tmp_mol_pdbqt_file.parent / f"{tmp_mol_pdbqt_file.stem}_out.pdbqt"
        )
        self.docked_sdf_file = self.docked_pdbqt_file.with_suffix(".sdf")

        commands: str = f"""{QVINA_PATH} \
            --receptor {receptor_pdbqt_file} \
            --ligand {tmp_mol_pdbqt_file} \
            --center_x {self.center[0]:.4f} \
            --center_y {self.center[1]:.4f} \
            --center_z {self.center[2]:.4f} \
            --size_x 20 --size_y 20 --size_z 20 \
            --exhaustiveness {exhaustiveness}
            obabel {self.docked_pdbqt_file} -O {self.docked_sdf_file} -h
        """

        self.proc = subprocess.Popen(
            "/bin/bash",
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.proc.stdin.write(commands.encode("utf-8"))
        self.proc.stdin.close()

        # return commands

    def run_sync(self, verbose=False, only_best_affinity=True):
        if not self.entered:
            raise RuntimeError("You must use this class with `with` statement.")

        self.run()
        while True:
            results = self.get_results()
            if results is not None:
                break

        assert isinstance(results, list)
        if len(results) == 0:
            raise Exception("No docking results found")

        if verbose:
            print(
                "Best affinity:", results[0]["affinity"]
            )  # , 'RMSD:', results[0]['rmsd_ref'])

        if only_best_affinity:
            return results[0]["affinity"]
        else:
            raise NotImplementedError()

    def get_results(self):
        if self.proc is None:  # Not started
            return None
        elif self.proc.poll() is None:  # In progress
            return None
        else:
            if self.output is None:
                self.output = self.proc.stdout.readlines()
                try:
                    self.results = parse_qvina_outputs(
                        self.docked_sdf_file, self.noH_rdmol
                    )
                except:
                    raise Exception(
                        self.docked_sdf_file.exists(), self.docked_pdbqt_file.exists()
                    )

            return self.results


if __name__ == "__main__":
    from tqdm import tqdm

    receptor_file = "dataset_outputs/mgd_case_study_default_val/GraphBP/0/receptor.pdb"
    ligand_files = list(Path(receptor_file).parent.glob("*.sdf"))
    ligand_files.sort()

    results = []
    for ligand_file in tqdm(ligand_files):
        with P2MDockingTask(receptor_file, ligand_file) as task:
            try:
                result = task.run_sync()
            except Exception as e:
                print("Error")
                print(e)
                result = None
            results.append(result)

    print(results)
    print([file.stem for file in ligand_files])
