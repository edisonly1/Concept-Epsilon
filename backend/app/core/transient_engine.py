from typing import Dict, Iterable, List

import numpy as np
import tsnet

from ..config import SimulationConfig
from .network_model import NetworkModel


class TransientEngine:
    """
    Thin wrapper around TSNet for running transient simulations
    and extracting time series at nodes.
    """

    def __init__(
        self,
        network_model: NetworkModel,
        sim_config: SimulationConfig | None = None,
    ) -> None:
        self.network_model = network_model
        self.sim_config = sim_config or network_model.sim_config

    def _base_transient_model(self) -> tsnet.network.TransientModel:
        """
        Build a TSNet TransientModel with wave speed and time options set,
        but *before* any events (valve closure, pump shutoff, leaks) or
        initialization.
        """
        tm = tsnet.network.TransientModel(str(self.network_model.inp_path))
        tm.set_wavespeed(self.sim_config.wave_speed)
        tm.set_time(self.sim_config.duration_seconds, self.sim_config.time_step)
        return tm

    def _initialize_and_run(
        self,
        tm: tsnet.network.TransientModel,
        engine: str | None = None,
    ) -> tsnet.network.TransientModel:
        """
        Run the WNTR steady-state solver to set initial conditions,
        then run the transient simulation.
        """
        t0 = 0.0
        engine = engine or self.sim_config.engine
        tm = tsnet.simulation.Initializer(tm, t0, engine)
        tm = tsnet.simulation.MOCSimulator(tm)
        return tm

    # ------------------------------------------------------------------
    # Baseline event simulations (no leak)
    # ------------------------------------------------------------------

    def simulate_pump_shutoff(
        self,
        pump_id: str,
        ts: float = 1.0,
        tc: float = 1.0,
        se: float = 0.0,
        m: int = 1,
    ) -> tsnet.network.TransientModel:
        tm = self._base_transient_model()
        pump_op = [tc, ts, se, m]
        tm.pump_shut_off(pump_id, pump_op)
        tm = self._initialize_and_run(tm)
        return tm

    def simulate_valve_closure(
        self,
        valve_id: str,
        ts: float = 5.0,
        tc: float = 1.0,
        se: float = 0.0,
        m: int = 2,
    ) -> tsnet.network.TransientModel:
        tm = self._base_transient_model()
        valve_op = [tc, ts, se, m]
        tm.valve_closure(valve_id, valve_op)
        tm = self._initialize_and_run(tm)
        return tm

    # ------------------------------------------------------------------
    # Valve closure + background leak
    # ------------------------------------------------------------------

    def simulate_valve_closure_with_leak(
        self,
        valve_id: str,
        leak_node_id: str,
        emitter_coeff: float = 0.01,
        ts: float = 5.0,
        tc: float = 1.0,
        se: float = 0.0,
        m: int = 2,
    ) -> tsnet.network.TransientModel:
        """
        Simulate a valve closure in a network that already has a background
        leak at `leak_node_id`.
        """
        tm = self._base_transient_model()

        tm.add_leak(leak_node_id, emitter_coeff)

        valve_op = [tc, ts, se, m]
        tm.valve_closure(valve_id, valve_op)

        tm = self._initialize_and_run(tm)
        return tm

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_node_heads(
        tm: tsnet.network.TransientModel,
        node_ids: Iterable[str],
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Extract head time series for given nodes from a completed TSNet simulation.

        Returns:
            dict[node_id] -> {"t": [...], "head": [...]}
        """
        tt = np.array(tm.simulation_timestamps, dtype=float).tolist()
        out: Dict[str, Dict[str, List[float]]] = {}

        for nid in node_ids:
            node = tm.get_node(nid)
            head_array = getattr(node, "head", getattr(node, "_head", None))
            if head_array is None:
                continue
            head = np.array(head_array, dtype=float).tolist()
            out[nid] = {"t": tt, "head": head}

        return out
