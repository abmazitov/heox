from ase import Atoms
from typing import List, Optional


class Pipeline:
    def __init__(
        self,
        atoms: Atoms,
        modules: List[object],
        modules_intervals: Optional[List[int]] = None,
    ):
        """
        Initialize the simulation pipeline.

        :param atoms: ASE Atoms object representing the system.
        :param modules: List of modules to be used in the pipeline.
        :param intervals: List of intervals for each module.
        """
        self.atoms = atoms
        self.modules = modules
        self.modules_intervals = (
            modules_intervals if modules_intervals is not None else [1] * len(modules)
        )

    def run(self, steps=50):
        """
        Run the hybrid Monte Carlo simulation.

        :param steps: Number of steps to run for each method.
        """
        for step in range(steps):
            for module, interval in zip(self.modules, self.intervals):
                if step % interval == 0:
                    module.run(1)
