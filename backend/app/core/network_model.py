from pathlib import Path
from typing import Optional

from ..config import NETWORKS_DIR, SimulationConfig


class NetworkModel:
    """
    Minimal wrapper around an EPANET network definition.

    Responsibilities:
    - Map a logical network name (e.g. 'Tnet2.inp') to a real file
      under backend/data/networks/.
    - Store simulation defaults (wave speed, duration).
    - Provide the .inp path for TSNet.
    """

    def __init__(
        self,
        inp_path: Path,
        sim_config: Optional[SimulationConfig] = None,
    ) -> None:
        self.inp_path = Path(inp_path)
        if not self.inp_path.exists():
            raise FileNotFoundError(f"EPANET INP not found: {self.inp_path}")

        self.sim_config: SimulationConfig = sim_config or SimulationConfig()
        self.name: str = self.inp_path.name

    @classmethod
    def from_name(
        cls,
        network_name: str,
        sim_config: Optional[SimulationConfig] = None,
    ) -> "NetworkModel":
        """
        Build a NetworkModel from a network_name like 'Tnet2.inp'.
        The corresponding file must live in backend/data/networks/.
        """
        inp_path = NETWORKS_DIR / network_name
        return cls(inp_path=inp_path, sim_config=sim_config)
