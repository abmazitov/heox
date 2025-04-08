from ase import Atoms
from ase.build import bulk, sort
from typing import Dict, Tuple, Optional
import numpy as np
from ase.spacegroup import crystal

ANION_ELEMENT = "O"
AVAILABLE_PATTERNS = ["rocksalt", "perovskite", "fluorite"]


def bulk_heo(
    pattern: str,
    cation_composition: Dict[str, float],
    a: float = 5.0,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    dopant: Optional[str] = None,
    dopant_fraction: Optional[float] = None,
) -> Atoms:
    """
    Create a bulk crystal structure for HEO materials given a crystal structure
    pattern and cation composition.

    :param pattern: The type of crystal structure (e.g., 'rocksalt', 'zincblende').
    :param cation_composition: A dictionary with cation species as keys and their
        stoichiometric coefficients as values.
    :param supercell: Optional supercell dimensions as a list of three integers.
        If None, a unit cell will be created.
    :return: An ASE Atoms object representing the bulk crystal structure.
    """
    # Rescale the cation composition to sum to 1
    cation_composition = {
        k: v / sum(cation_composition.values()) for k, v in cation_composition.items()
    }

    if dopant is not None:
        if dopant_fraction is None:
            raise ValueError(
                "If dopant is provided, dopant_fraction must also be provided."
            )
        # Add the dopant to the cation composition
        cation_composition[dopant] = dopant_fraction
        # Rescale the cation composition to sum to 1
        cation_composition = {
            k: v - dopant_fraction / len(cation_composition)
            for k, v in cation_composition.items()
        }

    # Generate the crystal pattern structure
    atoms = generate_base_crystal_pattern(pattern, a)
    if supercell is not None:
        atoms = atoms.repeat(supercell)

    anions_mask = [symbol == ANION_ELEMENT for symbol in atoms.get_chemical_symbols()]
    anion_sites = np.where(anions_mask)[0]
    cation_sites = np.delete(
        np.arange(len(atoms)),
        anion_sites,
    )
    num_cations_sites = len(cation_sites)

    for symbol, fraction in cation_composition.items():
        if not (num_cations_sites * fraction).is_integer():
            raise ValueError(
                f"Cannot create a bulk structure with {supercell} supercell and "
                f"{symbol} composition of {fraction}. The supercell yields {num_cations_sites} "
                "cation sites. Try either adjust the composition or use a different supercell."
            )
    # Calculate the new composition for the cations
    num_cations = {k: int(v * num_cations_sites) for k, v in cation_composition.items()}
    # Create a new list of atoms with the specified cation composition
    cation_symbols = np.repeat(
        list(cation_composition.keys()),
        list(num_cations.values()),
    )
    cation_symbols = np.random.permutation(cation_symbols)

    # Assign the cation symbols to the cation sites
    atoms.symbols[cation_sites] = cation_symbols
    # Assign the anion symbols to the anion sites
    atoms.symbols[anion_sites] = ANION_ELEMENT
    atoms = sort(atoms)
    return atoms


def generate_base_crystal_pattern(pattern: str, a: float = 5.0) -> Atoms:
    if pattern == "rocksalt":
        return bulk("MgO", "rocksalt", a=a, cubic=True)
    elif pattern == "perovskite":
        return crystal(
            ["Sr", "Ti", "O"],
            basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
            spacegroup=221,
            cellpar=[a, a, a, 90, 90, 90],
        )
    elif pattern == "fluorite":
        return bulk("CeO2", "fluorite", a=a, cubic=True)
    else:
        raise ValueError(
            f"Pattern '{pattern}' is not recognized. Available patterns: {AVAILABLE_PATTERNS}"
        )
