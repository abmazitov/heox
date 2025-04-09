from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
from metatrain.experimental.nativepet import NativePET

from heox.build import bulk_heo
from heox.mc import AtomSwapMonteCarlo, OnLatticeGCMC
from heox.pipeline import Pipeline
from heox.state import State


def test_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    atoms = bulk_heo(
        pattern="rocksalt",
        cation_composition={"Mg": 0.5, "Zn": 0.5},
        supercell=(2, 2, 2),
    )
    path = "https://huggingface.co/lab-cosmo/pet-mad/resolve/nativepet/models/pet-mad-latest.ckpt"
    if urlparse(path).scheme:
        path, _ = urlretrieve(path)
    NativePET.load_checkpoint(path).export().save(file="test.pt")
    calc = MetatensorCalculator(model="test.pt", device=device)
    atoms.calc = calc

    atoms.set_constraint(FixAtoms(mask=[True for atom in atoms]))
    fcf = FrechetCellFilter(atoms, hydrostatic_strain=True)

    optimizer = LBFGS(fcf, trajectory="relax.traj", logfile="relax.log")
    optimizer.run(fmax=0.05)

    atoms.constraints = []

    temperature = 600

    mc = AtomSwapMonteCarlo(
        temperature=temperature,
        types=["Mg", "Zn"],
    )

    vac_mc = AtomSwapMonteCarlo(
        temperature=temperature,
        types=["O", "X"],
    )

    gcmc = OnLatticeGCMC(
        temperature=temperature,
        chemical_potentials={"O": -1.0},
        use_relaxed_energies=False,
    )

    state = State.from_atoms(atoms)

    pipeline = Pipeline(
        state=state,
        modules=[mc, vac_mc, gcmc],
        log_options=["properties.step", "properties.energy"],
        loginterval=1,
        trajectory="test.xyz",
    )
    pipeline.run(20)
