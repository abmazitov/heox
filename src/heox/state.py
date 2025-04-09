from typing import Any, Dict, Optional

from ase import Atoms


class State:
    def __init__(self, system: Dict[str, Any], properties: Dict[str, Any]):
        self.system = system
        self.properties = properties

    @classmethod
    def from_atoms(cls, atoms: Atoms):
        """
        Create a State object from an ASE Atoms object.
        """
        system = {
            "positions": atoms.get_positions(),
            "types": atoms.get_chemical_symbols(),
            "cell": atoms.get_cell(),
            "pbc": atoms.get_pbc(),
            "calculator": atoms.calc,
        }
        properties = {
            "global_step": 0
            if atoms.info.get("global_step") is None
            else atoms.info["global_step"],
            "step": 0 if atoms.info.get("step") is None else atoms.info["step"],
            "temperature": atoms.get_temperature(),
            "energy": atoms.get_potential_energy() if atoms._calc is not None else None,
        }
        state = cls(system=system, properties=properties)
        return state

    def update(
        self,
        system: Optional[Dict[str, Any]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Update the state with new system and properties.
        """
        if system is not None:
            self.system.update(system)
        if properties is not None:
            self.properties.update(properties)

    def get(self, option: str):
        option_type, option_name = option.split(".")
        if option_type not in ["system", "properties"]:
            raise ValueError(
                f"Invalid option type: {option_type}. Must be 'system' or 'properties'."
            )
        if option_type == "system":
            if option_name not in self.system:
                raise ValueError(f"Invalid system option: {option_name}.")
            return self.system[option_name]
        elif option_type == "properties":
            if option_name not in self.properties:
                raise ValueError(f"Invalid properties option: {option_name}.")
            return self.properties[option_name]

    def get_atoms(self) -> Atoms:
        """
        Convert the current state to an ASE Atoms object.
        """
        atoms = Atoms(
            symbols=self.system["types"],
            positions=self.system["positions"],
            cell=self.system["cell"],
            pbc=self.system["pbc"],
        )
        if self.properties["temperature"] is not None:
            atoms.info["temperature"] = self.properties["temperature"]
        if self.properties["step"] is not None:
            atoms.info["step"] = self.properties["step"]
        if self.properties["energy"] is not None:
            atoms.info["energy"] = self.properties["energy"]
        return atoms

    def write_trajectory(self, filename: str):
        """
        Save the current state to a trajectory file.
        """
        atoms = self.get_atoms()
        atoms.write(filename, append=True)
