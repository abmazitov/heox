from ase import Atoms, Atom
import random
from typing import List, Dict
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class OnLatticeGrandCanonicalMonteCarlo:
    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        symbols: List[str],
        chemical_potentials: List[float],
        logging: bool = False,
        loginterval: int = 1,
    ):
        """
        Initialize the GrandCanonicalMonteCarlo simulation on lattice.

        :param atoms: ASE Atoms object representing the system.
        :param temperature: Simulation temperature in Kelvin.
        :param symbols: List of allowed chemical symbols for insertion/deletion.
        :param chemical_potentials: List of chemical potentials for each atomic type.
        """
        self.atoms = atoms
        self.temperature = temperature
        self.symbols = symbols
        self.chemical_potentials = chemical_potentials
        self.removal_allowed_indices: Dict[str, List[int]] = {
            symbol: [
                i for i, s in enumerate(atoms.get_chemical_symbols()) if s == symbol
            ]
            for symbol in symbols
        }
        self.insertion_allowed_atoms: Dict[str, List[Atom]] = {
            symbol: [] for symbol in symbols
        }
        self.step = 0
        self.accepted_insertions = 0
        self.rejected_insertions = 0
        self.accepted_removals = 0
        self.rejected_removals = 0

    def attemt_insertion(self):
        """
        Perform a random insertion of an atom into the lattice.
        """
        # Select a random atomic type based on chemical potentials
        atomic_type = random.choice(self.atomic_types)
        empty_sites = self.insertion_allowed_atoms[atomic_type]
        if not empty_sites:
            return False

        site = random.choice(empty_sites)

        initial_energy = self.atoms.get_potential_energy()

        self.atoms.append(site)

        final_energy = self.atoms.get_potential_energy()

        dE = final_energy - initial_energy

        # Metropolis criterion
        if not self.metropolis_criterion(dE):
            self.atoms.pop()
            self.rejected_insertions += 1
            return False
        else:
            self.insertion_allowed_atoms[atomic_type].remove(site)
            self.accepted_insertions += 1
            return True

    def attempt_removal(self):
        """
        Perform a random removal of an atom from the lattice.
        """
        # Select a random occupied site with an atomic type in the list
        atomic_type = random.choice(self.atomic_types)
        occupied_indices = self.removal_allowed_indices[atomic_type]
        if not occupied_indices:
            return False

        index = random.choice(occupied_indices)

        initial_energy = self.atoms.get_potential_energy()

        site = self.atoms.pop(index)

        final_energy = self.atoms.get_potential_energy()

        dE = final_energy - initial_energy

        # Metropolis criterion
        if not self.metropolis_criterion(
            dE, chemical_potential=self.chemical_potentials[atomic_type]
        ):
            # Revert the removal
            self.atoms.append(site)
            self.rejected_removals += 1
            return False
        else:
            self.removal_allowed_indices[atomic_type].remove(index)
            self.insertion_allowed_atoms[atomic_type].append(site)
            self.accepted_removals += 1
            return True

    def metropolis_criterion(self, delta_energy, chemical_potential):
        """
        Apply the Metropolis criterion to decide whether to accept a move.

        :param delta_energy: Energy change of the move.
        :param chemical_potential: Chemical potential of the atomic type.
        :return: True if the move is accepted, False otherwise.
        """
        boltzmann_factor = np.exp(
            -(delta_energy - chemical_potential) / (self.temperature)
        )
        return np.random.rand() < boltzmann_factor

    def run(self, steps=50):
        """
        Run the Grand Canonical Monte Carlo simulation for a given number of steps.

        :param steps: Number of Monte Carlo steps to perform.
        """
        for _ in range(steps):
            if np.random.rand() < 0.5:
                self._random_insertion()
            else:
                self._random_deletion()
            if self.logging and self.step % self.loginterval == 0:
                self.log()
            self.step += 1

    def log(self):
        e = self.atoms.get_potential_energy()
        T = time.localtime()
        name = self.__class__.__name__
        info = f"{name}: Step {self.step} | Time: {T.tm_hour}:{T.tm_min}:{T.tm_sec} | Energy: {e:.2f} eV"
        info += f" | Accepted Insertions: {self.accepted_insertions} | Rejected Insertions: {self.rejected_insertions}"
        info += f" | Accepted Removals: {self.accepted_removals} | Rejected Removals: {self.rejected_removals}"
        logger.info(info)
