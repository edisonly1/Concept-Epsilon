from typing import List, Literal, Optional

from pydantic import BaseModel


# ---- Health ----

class HealthResponse(BaseModel):
    status: Literal["ok", "error"] = "ok"


# ---- Time series primitives ----

class TimeSeries(BaseModel):
    t: List[float]
    values: List[float]


class NodeSeries(BaseModel):
    node_id: str
    head: TimeSeries


# ---- Simulation request/response ----

class SimulationRequest(BaseModel):
    """
    Request a transient simulation for a single event on a given network.

    network_name: name of the .inp file in backend/data/networks/
    event_type:   'pump_shut_off' or 'valve_closure'
    element_id:   pump ID or valve ID, matching the EPANET/TSNet ID
    node_ids:     which nodes to sample head at
    """

    network_name: str = "Tnet1.inp"
    event_type: Literal["pump_shut_off", "valve_closure"] = "valve_closure"
    element_id: str
    node_ids: List[str]
    duration_seconds: Optional[float] = None


class SimulationResponse(BaseModel):
    network_name: str
    event_type: str
    element_id: str
    node_series: List[NodeSeries]


# ---- Leak simulation & prediction requests ----

class LeakSimulationRequest(BaseModel):
    """
    Request a transient simulation for a valve-closure event with a
    background leak at a given junction.
    """

    network_name: str = "Tnet1.inp"
    event_type: Literal["valve_closure"] = "valve_closure"
    element_id: str = "VALVE"  # valve ID
    leak_node_id: str          # junction ID where leak is placed
    emitter_coeff: float = 0.01
    node_ids: List[str]
    duration_seconds: Optional[float] = None


class LeakPredictionRequest(BaseModel):
    """
    Request a leak prediction for a single transient event.

    For now this always runs a valve closure at `element_id` and, if
    leak_node_id is not None, injects a synthetic leak at that junction.
    """

    network_name: str = "Tnet1.inp"
    event_type: Literal["valve_closure"] = "valve_closure"
    element_id: str = "VALVE"
    leak_node_id: Optional[str] = None   # None => no-leak scenario
    emitter_coeff: float = 0.01
    duration_seconds: Optional[float] = None


# ---- Leak prediction outputs ----

class LeakCandidate(BaseModel):
    pipe_id: str
    model_prob: float
    physics_score: float
    combined_score: float


class LeakPrediction(BaseModel):
    leak_present: bool
    leak_probability: float
    top_candidates: List[LeakCandidate]
    uncertainty: float


class LeakPrediction(BaseModel):
    leak_present: bool
    leak_probability: float
    top_candidates: List[LeakCandidate]
    uncertainty: float
    # NEW FIELD:
    observed_data: List[NodeSeries] = []