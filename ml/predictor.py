from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ml.pinn_model import PhysicsInformedNN


DATA_PATH = Path(__file__).resolve().parent / "sandy_loam_nod.csv"
TRAIN_ITERATIONS = int(os.getenv("APRISOIL_PINN_ITERATIONS", "1000"))

_model: Optional["PhysicsInformedNN"] = None
_model_load_attempted = False


def get_model() -> Optional[PhysicsInformedNN]:
    global _model, _model_load_attempted

    if _model is not None:
        return _model
    if _model_load_attempted:
        return None

    _model_load_attempted = True
    try:
        from ml.pinn_model import DEFAULT_DATA_PATH, main_loop

        _model, _ = main_loop(
            hydrus="sandy_loam",
            depth_increment=1,
            noise=0,
            num_layers_psi=8,
            num_neurons_psi=40,
            num_layers_theta=1,
            num_neurons_theta=10,
            num_layers_K=1,
            num_neurons_K=10,
            number_random=111,
            data_path=DEFAULT_DATA_PATH,
            train_iterations=TRAIN_ITERATIONS,
            verbose=False,
        )
    except Exception:
        _model = None
    return _model


def _safe_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return numeric


def _deterministic_fallback(current_moisture: float, temperature: float) -> list[float]:
    moisture = float(np.clip(current_moisture, 0.0, 100.0))
    temp = float(np.clip(temperature, -20.0, 60.0))
    daily_loss = max(0.5, min(8.0, 1.2 + temp * 0.08))

    forecast: list[float] = []
    for day in range(1, 8):
        moisture = max(10.0, moisture - daily_loss - day * 0.08)
        forecast.append(round(moisture, 1))
    return forecast


def _normalize_forecast(values: list[float], current_moisture: float, temperature: float) -> list[float]:
    forecast: list[float] = []
    for value in values[:7]:
        numeric = _safe_float(value, current_moisture)
        forecast.append(round(float(np.clip(numeric, 0.0, 100.0)), 1))

    if len(forecast) < 7:
        fallback = _deterministic_fallback(current_moisture, temperature)
        forecast.extend(fallback[len(forecast) :])
    return forecast


def predict_moisture_7days(current_moisture: float, temperature: float) -> list[float]:
    current = _safe_float(current_moisture, 40.0)
    temp = _safe_float(temperature, 25.0)

    try:
        df = pd.read_csv(DATA_PATH)
        model = get_model()
        if model is None:
            return _deterministic_fallback(current, temp)

        t_star = df["time"].values[:, None]
        z_star = df["depth"].values[:, None]
        theta_pred, _ = model.predict(t_star, z_star)
        base_values = (theta_pred.flatten()[-7:] * 100.0).tolist()
        scaled_values = [value * (current / 40.0) for value in base_values]
        return _normalize_forecast(scaled_values, current, temp)
    except Exception:
        return _deterministic_fallback(current, temp)


def check_water_needed(forecast: list[float]) -> Optional[int]:
    for index, value in enumerate(forecast):
        if value < 30:
            return index + 1
    return None
