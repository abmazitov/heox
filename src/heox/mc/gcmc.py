import logging
import random
from typing import Dict, List

import numpy as np

from ..protocol import Protocol
from ..state import State
from ..utilities.potential_energy import (
    calculate_potential_energy,
    calculate_relaxed_potential_energy,
)


logger = logging.getLogger(__name__)


class OnLatticeGCMC(Protocol):
    def __init__(
        self,
        temperature: float,
        chemical_potentials: Dict[str, float],
        use_relaxed_energies: bool = False,
        invoke_every: int = 1,
        steps_per_invoke: int = 1,
    ):
        """
        Initialize the on-lattice Grand-Canonical Monte-Carlo protocol.

        :param temperature: Simulation temperature in Kelvin.
        :param chemical_potentials: Dict of chemical potentials for each atomic type.
        :param invoke_every: Invoke this protocol every n steps.
        :param steps_per_invoke: Number of steps to run for each method.
        """
        super().__init__(invoke_every, steps_per_invoke)
        self.temperature = temperature
        self.chemical_potentials = chemical_potentials
        self.types = list(chemical_potentials.keys())
        self.use_relaxed_energies = use_relaxed_energies

        self.accepted_insertions = 0
        self.rejected_insertions = 0
        self.accepted_removals = 0
        self.rejected_removals = 0

    def initialize(
        self,
        state: State,
    ):
        """
        Initialize the simulation with the given state.
        """
        state_types = set(state.system["types"])
        if not set(self.types).issubset(state_types):
            raise ValueError(
                "The provided types are not a subset of the system's types."
            )
        self.removed_indices: List[int] = []

    def step(self, state: State):
        if np.random.rand() < 0.5:
            self._attempt_insertion(state)
        else:
            self._attempt_removal(state)
        self.num_invokes += 1

    def _attempt_removal(self, state: State):
        """
        Perform a random removal of an atom from the lattice.
        """
        # Step 1. Select a random occupied site with a selected atomic type
        types = state.get("system.types")
        atomic_type = random.choice(self.types)
        allowed_indices: List[int] = [
            i for i, s in enumerate(types) if s == atomic_type
        ]
        if not allowed_indices:
            self.rejected_removals += 1
            return

        index = random.choice(allowed_indices)

        # Step 2. Attempt to remove the atom from the lattice

        initial_energy = state.get("properties.energy")
        if initial_energy is None:
            initial_energy = calculate_potential_energy(state)

        attempt_types = types.copy()
        attempt_types[index] = "X"  # Placeholder for empty site
        state.update(system={"types": attempt_types})

        if self.use_relaxed_energies:
            final_energy = calculate_relaxed_potential_energy(state)
        else:
            final_energy = calculate_potential_energy(state)

        dE = final_energy - initial_energy

        # Step 3. Checking if the removal is accepted or rejected
        if not self._metropolis_criterion(
            dE, chemical_potential=self.chemical_potentials[atomic_type]
        ):
            # Revert the removal
            assert attempt_types != types, "The swap should have changed the types."
            state.update(system={"types": types}, properties={"energy": initial_energy})
            self.rejected_removals += 1
        else:
            state.update(properties={"energy": final_energy})
            self.removed_indices.append(index)
            self.accepted_removals += 1

    def _attempt_insertion(self, state: State):
        """
        Perform a random insertion of an atom into the lattice.
        """
        """
        Perform a random removal of an atom from the lattice.
        """
        # Step 1. Select a random type to insert
        types = state.get("system.types")
        atomic_type = random.choice(self.types)
        allowed_indices = self.removed_indices
        if not allowed_indices:
            self.rejected_insertions += 1
            return

        index = random.choice(allowed_indices)

        # Step 2. Attempt to add the atom to the lattice

        initial_energy = state.get("properties.energy")
        if initial_energy is None:
            initial_energy = calculate_potential_energy(state)

        attempt_types = types.copy()
        attempt_types[index] = atomic_type
        state.update(system={"types": attempt_types})

        if self.use_relaxed_energies:
            final_energy = calculate_relaxed_potential_energy(state)
        else:
            final_energy = calculate_potential_energy(state)

        dE = final_energy - initial_energy

        # Step 3. Checking if the insertion is accepted or rejected
        if not self._metropolis_criterion(
            dE, chemical_potential=self.chemical_potentials[atomic_type]
        ):
            # Revert the removal
            assert attempt_types != types, "The swap should have changed the types."
            state.update(system={"types": types}, properties={"energy": initial_energy})
            self.rejected_insertions += 1
        else:
            state.update(properties={"energy": final_energy})
            self.removed_indices.remove(index)
            self.accepted_insertions += 1

    def _metropolis_criterion(self, delta_energy, chemical_potential):
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
