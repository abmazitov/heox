from ase.optimize import LBFGS

from ..state import State


def calculate_potential_energy(state: State):
    """
    Calculate the potential energy of the system.
    :param state: The current state of the system.
    :return: The potential energy of the system.
    """

    atoms = state.get_atoms()
    atoms.calc = state.get("system.calculator")
    # Check for empty sites
    empty_sites = [
        i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "X"
    ]
    del atoms[empty_sites]
    potential_energy = atoms.get_potential_energy()
    return potential_energy


def calculate_relaxed_potential_energy(state: State, fmax: float = 0.05):
    """
    Calculate the potential energy of the system after relaxation.
    :param state: The current state of the system.
    :param fmax: The maximum force for convergence.
    :return: The potential energy of the system.
    """
    atoms = state.get_atoms()
    atoms.calc = state.get("system.calculator")
    # Check for empty sites
    empty_sites = [
        i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "X"
    ]
    del atoms[empty_sites]

    optimizer = LBFGS(atoms, trajectory="tmp.traj", logfile="tmp.log")
    optimizer.run(fmax=fmax)
    potential_energy = atoms.get_potential_energy()
    return potential_energy
