import logging
import time
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.units import kB


logger = logging.getLogger(__name__)


class AtomSwapMonteCarlo:
    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        symbols: List[str],
        logging: bool = False,
        loginterval: int = 1,
        trajectory: Optional[str] = None,
    ):
        """
        Initialize the Monte Carlo simulation.

        :param atoms: ASE Atoms object representing the system.
        :param temperature: Simulation temperature in Kelvin.
        :param symbols: List of allowed chemical symbols for swapping.
        """
        self.atoms = atoms
        self.temperature = temperature
        self.symbols = symbols
        if len(symbols) < 2:
            raise ValueError("At least two symbols are required for swapping.")
        self._allowed_indices = [
            i
            for i, symbol in enumerate(atoms.get_chemical_symbols())
            if symbol in symbols
        ]
        self.step = 0
        self.accepted = 0
        self.rejected = 0
        self.logging = logging
        self.loginterval = loginterval

    def run(self, steps=50):
        """
        Run the Monte Carlo simulation.

        :param steps: Number of Monte Carlo steps to perform.
        """
        for _ in range(steps):
            self.attempt_atomswap()
            self.step += 1
            if self.logging and self.step % self.loginterval == 0:
                self.log()

    def log(self):
        e = self.atoms.get_potential_energy()
        T = time.localtime()
        name = self.__class__.__name__
        info = f"{name}: Step {self.step} | Time: {T.tm_hour}:{T.tm_min}:{T.tm_sec} | "
        info += f"Energy: {e:.2f} eV | "
        info += f"Accepted: {self.accepted} | Rejected: {self.rejected}"
        logger.info(info)
        if self.trajectory is not None:
            self.atoms.write(self.trajectory, append=True)

    def attempt_atomswap(self):
        """
        Attempt to swap two atoms in the lattice.
        """
        num_attepts = 100
        symbols = self.atoms.get_chemical_symbols()
        while num_attepts > 0:
            atom_index_1, atom_index_2 = np.random.choice(
                self._allowed_indices, size=2, replace=False
            )
            symbol_1, symbol_2 = symbols[atom_index_1], symbols[atom_index_2]
            if symbol_1 != symbol_2:
                break
            num_attepts -= 1
        if num_attepts == 0:
            raise RuntimeError(
                "Could not find two different atoms to swap after 100 attempts."
            )

        # Calculate the initial energy
        initial_energy = self.atoms.get_potential_energy()

        # Swap the chemical symbols
        self.atoms[atom_index_1].symbol, self.atoms[atom_index_2].symbol = (
            symbol_2,
            symbol_1,
        )
        final_energy = self.atoms.get_potential_energy()

        # Calculate energy change
        dE = final_energy - initial_energy

        # Metropolis criterion
        if not self.metropolis_criterion(dE):
            # Revert the swap
            self.atoms[atom_index_1].symbol, self.atoms[atom_index_2].symbol = (
                symbol_1,
                symbol_2,
            )
            self.rejected += 1
        else:
            self.accepted += 1

    def metropolis_criterion(self, dE):
        """
        Decide whether to accept the proposed move based on the Metropolis criterion.
        """
        if dE < 0:
            return True
        else:
            probability = np.exp(-dE / (kB * self.temperature))
            return np.random.rand() < probability
