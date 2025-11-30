from fastapi import APIRouter, HTTPException
import numpy as np

from ..config import SimulationConfig
from ..core.network_model import NetworkModel
from ..core.transient_engine import TransientEngine
from ..core.leak_localizer import LeakLocalizer
from ..core.schemas import (
    SimulationRequest,
    SimulationResponse,
    NodeSeries,
    TimeSeries,
    LeakSimulationRequest,
    LeakPredictionRequest,
    LeakCandidate,
    LeakPrediction,
)

router = APIRouter(prefix="/events", tags=["events"])

# Single global model instance
leak_localizer = LeakLocalizer()



# Basic transient simulation (no leak)

@router.post("/simulate", response_model=SimulationResponse)
def simulate_event(req: SimulationRequest) -> SimulationResponse:
    """
    Run a transient simulation for a single event and return
    node head time series.
    """

    sim_cfg = SimulationConfig(
        duration_seconds=(
            req.duration_seconds
            if req.duration_seconds is not None
            else SimulationConfig().duration_seconds
        )
    )

    try:
        network = NetworkModel.from_name(req.network_name, sim_config=sim_cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    engine = TransientEngine(network_model=network, sim_config=sim_cfg)

    try:
        if req.event_type == "pump_shut_off":
            tm = engine.simulate_pump_shutoff(pump_id=req.element_id)
        elif req.event_type == "valve_closure":
            tm = engine.simulate_valve_closure(valve_id=req.element_id)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported event_type: {req.event_type}",
            )
    except KeyError:
        # Bad pump/valve ID
        raise HTTPException(
            status_code=400,
            detail=(
                f"Element ID '{req.element_id}' not found in network "
                f"'{req.network_name}'. Check the ID against the INP file."
            ),
        )

    series_dict = engine.extract_node_heads(tm, req.node_ids)

    node_series = [
        NodeSeries(
            node_id=nid,
            head=TimeSeries(
                t=series_dict[nid]["t"],
                values=series_dict[nid]["head"],
            ),
        )
        for nid in req.node_ids
        if nid in series_dict
    ]

    if not node_series:
        raise HTTPException(
            status_code=400,
            detail="No node series could be extracted. Check node_ids against network.",
        )

    return SimulationResponse(
        network_name=req.network_name,
        event_type=req.event_type,
        element_id=req.element_id,
        node_series=node_series,
    )



# Transient simulation with background leak
@router.post("/simulate_leak", response_model=SimulationResponse)
def simulate_leak(req: LeakSimulationRequest) -> SimulationResponse:
    """
    Run a transient simulation for a valve closure with a background leak
    at `leak_node_id`.
    """

    if req.event_type != "valve_closure":
        raise HTTPException(
            status_code=400,
            detail="Only valve_closure is supported in simulate_leak for now.",
        )

    sim_cfg = SimulationConfig(
        duration_seconds=(
            req.duration_seconds
            if req.duration_seconds is not None
            else SimulationConfig().duration_seconds
        )
    )

    try:
        network = NetworkModel.from_name(req.network_name, sim_config=sim_cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    engine = TransientEngine(network_model=network, sim_config=sim_cfg)

    try:
        tm = engine.simulate_valve_closure_with_leak(
            valve_id=req.element_id,
            leak_node_id=req.leak_node_id,
            emitter_coeff=req.emitter_coeff,
        )
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Element ID '{req.element_id}' or leak node "
                f"'{req.leak_node_id}' not found in network '{req.network_name}'."
            ),
        )

    series_dict = engine.extract_node_heads(tm, req.node_ids)

    node_series = [
        NodeSeries(
            node_id=nid,
            head=TimeSeries(
                t=series_dict[nid]["t"],
                values=series_dict[nid]["head"],
            ),
        )
        for nid in req.node_ids
        if nid in series_dict
    ]

    if not node_series:
        raise HTTPException(
            status_code=400,
            detail="No node series could be extracted. Check node_ids against network.",
        )

    return SimulationResponse(
        network_name=req.network_name,
        event_type=req.event_type,
        element_id=req.element_id,
        node_series=node_series,
    )



# Leak prediction endpoint
@router.post("/predict_leak", response_model=LeakPrediction)
def predict_leak(req: LeakPredictionRequest) -> LeakPrediction:
    """
    Run a valve-closure transient simulation (with optional synthetic leak),
    feed the resulting head traces at the model's sensors through the trained
    CNN leak localizer, and return leak probabilities fused with a
    physics-consistency score.

    Fusion:
      - Start from ML softmax probs.
      - For each leak class AND the 'none' class, compute physics_score in [0, 1].
      - Reweight each class prob by (alpha + (1-alpha)*physics_score).
      - Renormalize to get fused probabilities p_fused.
    """

    if leak_localizer.model is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Leak localization model not loaded. "
                "Train it with backend/ml/train_cnn_leak_localizer.py first."
            ),
        )

    if req.event_type != "valve_closure":
        raise HTTPException(
            status_code=400,
            detail="Only valve_closure is supported in predict_leak for now.",
        )

    sim_cfg = SimulationConfig(
        duration_seconds=(
            req.duration_seconds
            if req.duration_seconds is not None
            else SimulationConfig().duration_seconds
        )
    )

    try:
        network = NetworkModel.from_name(req.network_name, sim_config=sim_cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    engine = TransientEngine(network_model=network, sim_config=sim_cfg)

    # ------------------------------------------------------------------
    # 1. Simulate observed event (with or without leak)
    # ------------------------------------------------------------------
    try:
        if req.leak_node_id is None:
            tm_obs = engine.simulate_valve_closure(valve_id=req.element_id)
        else:
            tm_obs = engine.simulate_valve_closure_with_leak(
                valve_id=req.element_id,
                leak_node_id=req.leak_node_id,
                emitter_coeff=req.emitter_coeff,
            )
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Element ID '{req.element_id}' or leak node "
                f"'{req.leak_node_id}' not found in network '{req.network_name}'."
            ),
        )

    sensors = leak_localizer.sensors
    if not sensors:
        raise HTTPException(
            status_code=500,
            detail="Model has no sensor metadata; re-train and save with sensors.",
        )

    series_obs = engine.extract_node_heads(tm_obs, sensors)

    time_grid = leak_localizer.time_grid
    if not time_grid:
        some_node = next(iter(series_obs))
        time_grid = series_obs[some_node]["t"]

    T = len(time_grid)
    C = len(sensors)

    def build_tensor_from_series(series_dict: dict[str, dict[str, list[float]]]) -> np.ndarray:
        X_local = np.zeros((1, C, T), dtype=np.float32)
        for ci, sid in enumerate(sensors):
            if sid not in series_dict:
                continue
            t_list = series_dict[sid]["t"]
            values = series_dict[sid]["head"]

            if len(t_list) == T and all(
                abs(float(t_list[i]) - float(time_grid[i])) < 1e-6 for i in range(T)
            ):
                X_local[0, ci, :] = np.array(values, dtype=np.float32)
            else:
                lookup = {float(t): float(v) for t, v in zip(t_list, values)}
                X_local[0, ci, :] = np.array(
                    [lookup.get(float(t), 0.0) for t in time_grid], dtype=np.float32
                )

        return X_local

    # Observed tensor
    X_obs = build_tensor_from_series(series_obs)

   
    # 2. ML prediction on observed traces
    probs = leak_localizer.predict_proba(X_obs)[0]  # [num_classes]
    num_classes = probs.shape[0]

    class_mapping = leak_localizer.class_mapping or {}
    idx_to_class = leak_localizer.idx_to_class or {}

    if "none" not in class_mapping:
        raise HTTPException(
            status_code=500,
            detail="Model class mapping does not contain 'none' class.",
        )

    none_idx = class_mapping["none"]
    leak_indices = [i for i in range(num_classes) if i != none_idx]

    
    # 3. Physics scores
    
    physics_scores: dict[int, float] = {}

    
    try:
        tm_none = engine.simulate_valve_closure(valve_id=req.element_id)
        series_none = engine.extract_node_heads(tm_none, sensors)
        X_none = build_tensor_from_series(series_none)

        mse_none = float(np.mean((X_obs - X_none) ** 2))
        denom = float(np.var(X_obs) + 1e-6)
        physics_score_none = max(0.0, 1.0 - mse_none / denom)
    except Exception:
        physics_score_none = 0.0

    physics_scores[none_idx] = physics_score_none

    # Physics scores for each leak candidate
    for i in leak_indices:
        name = idx_to_class.get(i, f"class_{i}")
        try:
            tm_cand = engine.simulate_valve_closure_with_leak(
                valve_id=req.element_id,
                leak_node_id=name,
                emitter_coeff=req.emitter_coeff,
            )
            series_cand = engine.extract_node_heads(tm_cand, sensors)
            X_cand = build_tensor_from_series(series_cand)

            mse = float(np.mean((X_obs - X_cand) ** 2))
            denom = float(np.var(X_obs) + 1e-6)
            physics_score = max(0.0, 1.0 - mse / denom)
        except KeyError:
            physics_score = 0.0

        physics_scores[i] = physics_score

    
    # ML probabilities with physics scores
    
    alpha = 0.2  # minimum physics influence

    p_unnorm = np.zeros_like(probs)

    for idx in range(num_classes):
        phys = physics_scores.get(idx, 0.0)
        w = alpha + (1.0 - alpha) * phys  # in [alpha, 1]
        p_unnorm[idx] = float(probs[idx]) * w

    Z = float(p_unnorm.sum())
    if Z <= 0:
        p_fused = np.full_like(probs, 1.0 / num_classes)
    else:
        p_fused = p_unnorm / Z

    leak_probability = float(1.0 - p_fused[none_idx])

    # More sensitive detector: treat anything above 0.3 as a leak,
    leak_present = leak_probability >= 0.3

    # Uncertainty from distribution
    eps = 1e-8
    entropy = float(-(p_fused * np.log(p_fused + eps)).sum())
    max_entropy = float(np.log(num_classes))
    uncertainty = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # 5. Build ranked candidates using fused probabilities
    k = min(3, len(leak_indices))
    leak_indices_sorted = sorted(
        leak_indices, key=lambda i: float(p_fused[i]), reverse=True
    )[:k]

    top_candidates: list[LeakCandidate] = []
    for i in leak_indices_sorted:
        name = idx_to_class.get(i, f"class_{i}")
        top_candidates.append(
            LeakCandidate(
                pipe_id=name,
                model_prob=float(probs[i]),
                physics_score=float(physics_scores.get(i, 0.0)),
                combined_score=float(p_fused[i]),
            )
        )

    formatted_obs_data = [
        NodeSeries(
            node_id=nid,
            head=TimeSeries(
                t=series_obs[nid]["t"],
                values=series_obs[nid]["head"],
            ),
        )
        for nid in sensors
        if nid in series_obs
    ]

    return LeakPrediction(
        leak_present=leak_present,
        leak_probability=leak_probability,
        top_candidates=top_candidates,
        uncertainty=uncertainty,
        observed_data=formatted_obs_data 
    )
