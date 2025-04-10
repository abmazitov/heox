import logging
from typing import List

import numpy as np
from ase.units import kB

from ..protocol import Protocol
from ..state import State
from ..utilities.potential_energy import calculate_potential_energy


logger = logging.getLogger(__name__)


class AtomSwapMonteCarlo(Protocol):
    def __init__(
        self,
        name: str,
        temperature: float,
        types: List[str],
        invoke_every: int = 1,
        steps_per_invoke: int = 1,
    ):
        """
        Initialize the Atom-Swap Monte-Carlo protocol.

        :param temperature: Simulation temperature in Kelvin.
        :param types: List of allowed atomic types for swapping.
        :param invoke_every: Invoke this protocol every n steps.
        :param steps_per_invoke: Number of steps to run for each method.

        """
        super().__init__(name, invoke_every, steps_per_invoke)
        self.temperature = temperature
        self.types = types
        if len(types) < 2:
            raise ValueError("At least two atom types are required for swapping.")
        self.accepted_swaps = 0
        self.rejected_swaps = 0

    def initialize(
        self,
        state: State,
    ):
        """
        Initialize the simulation with the given state.
        """
        state_types = set(state.get("system.types") + ["X"])
        if not set(self.types).issubset(state_types):
            raise ValueError(
                "The provided types are not a subset of the system's types."
            )

    def step(self, state: State):
        self._attempt_atomswap(state)
        self.num_invokes += 1
        state.update(
            modules={
                "mc.asmc": {
                    "accepted_swaps": self.accepted_swaps,
                    "rejected_swaps": self.rejected_swaps,
                }
            }
        )

    def _attempt_atomswap(self, state: State):
        """
        Attempt to swap two atoms in the lattice.
        """
        num_attepts = 100

        # Step 1. Choosing a pair of atom to swap
        types = state.get("system.types")
        allowed_indices = [i for i, symbol in enumerate(types) if symbol in self.types]
        while num_attepts > 0:
            atom_index_1, atom_index_2 = np.random.choice(
                allowed_indices, size=2, replace=False
            )
            symbol_1, symbol_2 = types[atom_index_1], types[atom_index_2]
            if symbol_1 != symbol_2:
                break
            num_attepts -= 1
        if num_attepts == 0:
            self.rejected_swaps += 1
            return

        # Step 2. Calculating the energy change upon swapping
        initial_energy = state.get("properties.energy")
        if initial_energy is None:
            initial_energy = calculate_potential_energy(state)

        attempt_types = types.copy()
        attempt_types[atom_index_1], attempt_types[atom_index_2] = symbol_2, symbol_1
        state.update(system={"types": attempt_types})

        final_energy = calculate_potential_energy(state)

        dE = final_energy - initial_energy

        # Step 3. Checking if the swap is accepted or rejected
        if not self._metropolis_criterion(dE):
            assert attempt_types != types, "The swap should have changed the types."
            state.update(system={"types": types}, properties={"energy": initial_energy})
            self.rejected_swaps += 1
        else:
            state.update(properties={"energy": final_energy})
            self.accepted_swaps += 1

    def _metropolis_criterion(self, delta_energy: float) -> bool:
        """
        Decide whether to accept the proposed move based on the Metropolis criterion.
        """
        if delta_energy < 0:
            return True
        else:
            probability = np.exp(-delta_energy / (kB * self.temperature))
            return np.random.rand() < probability

    def _get_log_options(self):
        return {
            f"module.{self.name}.accepted_swaps": self.accepted_swaps,
            f"module.{self.name}.rejected_swaps": self.rejected_swaps,
        }
