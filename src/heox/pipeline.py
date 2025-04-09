import logging
import os
from typing import List, Optional

from .state import State


logger = logging.getLogger(__name__)

LOGGING_OPTIONS_DICT = {
    "properties.step": "Step",
    "properties.energy": "Potential Energy",
    "properties.temperature": "Temperature",
}


class Pipeline:
    def __init__(
        self,
        state: State,
        modules: List[object],
        log_options: Optional[List[str]] = None,
        loginterval: Optional[int] = None,
        trajectory: Optional[str] = None,
    ):
        """
        Initialize the simulation pipeline.

        :param state: State to be used in the pipeline.
        :param modules: List of modules to be executed in the pipeline.
        :param log_options: List of logging options.
        :param loginterval: Interval for logging.
        :param trajectory: Path to save the trajectory.
        """
        self.state = state
        self.modules = modules
        if log_options is not None:
            for logname in log_options:
                if logname not in LOGGING_OPTIONS_DICT:
                    raise ValueError(
                        f"Invalid logging name: {logname}. "
                        f"Available options are: {list(LOGGING_OPTIONS_DICT.keys())}"
                    )
        self.log_options = log_options
        self.loginterval = loginterval if loginterval else 1
        self.trajectory = trajectory
        self.initialize()

    def _print_logging_header(self):
        """
        Print the initial logging line.
        """
        header = "HEO-X: A Python package for hybrid Monte Carlo simulations of HEOs\n"
        header += "\t".join([LOGGING_OPTIONS_DICT[opt] for opt in self.log_options])
        logger.info(header)

    def initialize(self):
        """
        Initialize the pipeline by setting up the modules.
        """
        for module in self.modules:
            module.initialize(self.state)
        if self.log_options:
            self._print_logging_header()
        if self.trajectory is not None:
            if os.path.exists(os.path.abspath(self.trajectory)):
                os.remove(self.trajectory)

    def run(self, steps=50):
        """
        Run the hybrid Monte Carlo simulation.

        :param steps: Number of steps to run for each method.
        """
        for step in range(steps):
            for module in self.modules:
                module.evolve(self.state)
            if self.log_options and step % self.loginterval == 0:
                self.log()

    def log(self):
        """
        Log the current state of the simulation.
        """
        info = "\t".join([str(self.state.get(opt)) for opt in self.log_options])
        logger.info(info)
        if self.trajectory is not None:
            self.state.write_trajectory(self.trajectory)
