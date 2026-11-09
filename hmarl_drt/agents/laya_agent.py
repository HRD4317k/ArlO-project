"""Laya System-One agent adapter for the turbine-control stack.

Laya returns typed choices, not free-form text or continuous control values.
This adapter asks one multi-question decision about each observation, then
maps its selected action label to a bounded action vector understood by the
environment. A deterministic safety check remains authoritative for critical
limits; Laya is a decision proposal, not a safety-certified controller.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import requests
import torch
import torch.nn as nn

DEFAULT_LIMITS = {"omega_min": 0.5, "omega_max": 1.5, "T_max": 100.0, "delta_crit": 0.9}

# Action vectors are exactly what the legacy FMAgent.act returned.
ACTION_TEMPLATES: dict[str, tuple[float, float, float]] = {
    "nominal": (0.0, 0.0, -1.0),
    "high_wind_pitch": (0.7, 0.0, -1.0),
    "thermal_derate": (0.3, -0.5, -1.0),
    "degradation_derate": (0.5, -0.3, 1.0),
}

# Weights are exactly the legacy FMAgent.get_reward_weights heuristic cases.
REWARD_PROFILES: dict[str, dict[str, float]] = {
    "balanced": {"energy": 0.30, "rul": 0.25, "safety": 0.20, "physics": 0.15, "curiosity": 0.10},
    "safety_priority": {"energy": 0.10, "rul": 0.20, "safety": 0.50, "physics": 0.15, "curiosity": 0.05},
    "lifetime_priority": {"energy": 0.10, "rul": 0.60, "safety": 0.20, "physics": 0.05, "curiosity": 0.05},
    "power_priority": {"energy": 0.50, "rul": 0.20, "safety": 0.15, "physics": 0.10, "curiosity": 0.05},
}


def _rounded(values) -> tuple[float, ...]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)[:8]
    return (round(v[0], 2), round(v[1], 1), round(v[2], 2), round(v[3], 1),
            round(v[4], 2), round(v[5], 3), round(v[6], 2), round(v[7], 2))


def turbine_state_text(values) -> str:
    """Observation as the text Laya reads; used identically for training and inference."""
    om, wind, tq, temp, pw, delta, grid, wx = _rounded(values)
    return (f"Wind turbine reading: rotor speed {om:.2f} rad/s; wind speed {wind:.1f} m/s; "
            f"generator torque {tq:.2f}; bearing temperature {temp:.1f} C; power output {pw:.2f}; "
            f"degradation index {delta:.3f}; grid demand {grid:.2f}; weather index {wx:.2f}.")


def teacher_labels(values, omega_min=0.5, omega_max=1.5, T_max=100.0, delta_crit=0.9) -> dict[str, str]:
    """Legacy FMAgent decision logic (assess / act / get_reward_weights), on the rounded state."""
    omega, wind, _, temp, _, delta, grid, _ = _rounded(values)
    if delta > delta_crit - 0.1 or temp > 0.95 * T_max:
        severity = "critical"
    elif temp > 0.85 * T_max or omega > omega_max or omega < omega_min:
        severity = "warning"
    else:
        severity = "normal"
    if delta > 0.7:
        action = "degradation_derate"
    elif temp > 80.0:
        action = "thermal_derate"
    elif wind > 20.0:
        action = "high_wind_pitch"
    else:
        action = "nominal"
    if delta > 0.7:
        profile = "lifetime_priority"
    elif temp > 85.0:
        profile = "safety_priority"
    elif grid > 0.8:
        profile = "power_priority"
    else:
        profile = "balanced"
    return {"severity": severity, "action": action, "reward_profile": profile}


def build_questions(omega_min=0.5, omega_max=1.5, T_max=100.0, delta_crit=0.9) -> dict[str, dict[str, Any]]:
    crit_delta, crit_temp, warn_temp = delta_crit - 0.1, 0.95 * T_max, 0.85 * T_max
    return {
        "severity": {
            "type": "choice",
            "instructions": "Grade the wind turbine operating condition. Critical overrides warning.",
            "criteria": {
                "normal": "bearing temperature at most %g C, rotor speed %g to %g rad/s, degradation at most %g"
                          % (warn_temp, omega_min, omega_max, crit_delta),
                "warning": "bearing temperature above %g C, or rotor speed outside %g to %g rad/s"
                           % (warn_temp, omega_min, omega_max),
                "critical": "bearing temperature above %g C, or degradation index above %g"
                            % (crit_temp, crit_delta),
            },
        },
        "action": {
            "type": "choice",
            "instructions": "Choose the control action for this wind turbine reading.",
            "criteria": {
                "nominal": "default settings, no maintenance",
                "high_wind_pitch": "wind speed above 20 m/s, pitch up for protection",
                "thermal_derate": "bearing temperature above 80 C, pitch up and cut torque",
                "degradation_derate": "degradation index above 0.7, derate and request maintenance",
            },
        },
        "reward_profile": {
            "type": "choice",
            "instructions": "Choose the reward priority profile for this wind turbine reading.",
            "criteria": {
                "balanced": "none of the other conditions hold",
                "safety_priority": "bearing temperature above 85 C",
                "lifetime_priority": "degradation index above 0.7",
                "power_priority": "grid demand above 0.8",
            },
        },
    }


class LayaAgent:
    """System-One decision agent with the method surface expected by training.

    The inference service is intentionally separate from the training Python
    environment. This keeps the Laya/CUDA/Transformers dependency stack from
    changing the existing SAC environment and permits the model to run on the
    NVIDIA host behind a local HTTP endpoint.
    """

    STATE_LABELS = (
        "rotor_speed_rad_s",
        "wind_speed_m_s",
        "generator_torque_normalized",
        "bearing_temperature_c",
        "power_output_normalized",
        "degradation_index",
        "grid_demand_normalized",
        "weather_feature_normalized",
    )

    ACTIONS = ACTION_TEMPLATES
    REWARD_PROFILES = REWARD_PROFILES

    def __init__(
        self,
        state_dim: int = 8,
        device: str = "cpu",
        endpoint: str | None = None,
        model_name: str = "english",
        api_key: str | None = None,
        timeout_s: float = 10.0,
        omega_min: float = 0.5,
        omega_max: float = 1.5,
        T_max: float = 100.0,
        delta_crit: float = 0.9,
        session: requests.Session | None = None,
    ) -> None:
        self.device = device
        self.endpoint = (
            endpoint or os.environ.get(
                "LAYA_BASE_URL", "http://127.0.0.1:8000/v1/systemone"
            )
        ).rstrip("/")
        self.model_name = model_name
        self.api_key = (api_key or os.environ.get("LAYA_API_KEY", "")).strip()
        self.timeout_s = float(timeout_s)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.T_max = float(T_max)
        self.delta_crit = float(delta_crit)
        self._session = session or requests.Session()
        self._cached_state: bytes | None = None
        self._cached_decision: dict[str, Any] | None = None
        self.context_history: list[dict[str, Any]] = []
        self.max_context = 10

        # Preserve the 128-dimensional FM-slot interface used by the shared
        # embedding module. Decisions themselves come from the Laya service.
        self.embed_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
        ).to(device)

    def _state_key(self, state: np.ndarray) -> tuple[np.ndarray, bytes]:
        values = np.asarray(state, dtype=np.float32).reshape(-1)
        if values.size < len(self.STATE_LABELS):
            raise ValueError(
                f"Laya expects {len(self.STATE_LABELS)} observation values; got {values.size}"
            )
        return values, values.tobytes()

    def _questions(self) -> dict[str, dict[str, Any]]:
        return build_questions(self.omega_min, self.omega_max, self.T_max, self.delta_crit)

    def _get_decision(self, state: np.ndarray) -> dict[str, Any]:
        values, key = self._state_key(state)
        if self._cached_state == key and self._cached_decision is not None:
            return self._cached_decision

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model_name,
            "state": turbine_state_text(values),
            "questions": self._questions(),
        }
        try:
            response = self._session.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Laya inference request failed at {self.endpoint}: {exc}") from exc
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Laya returned invalid JSON") from exc

        answers = body.get("answers") if isinstance(body, dict) else None
        if not isinstance(answers, dict):
            raise RuntimeError("Laya response is missing the 'answers' object")
        decision: dict[str, Any] = {}
        for field in ("severity", "action", "reward_profile"):
            answer = answers.get(field)
            if not isinstance(answer, dict) or not isinstance(answer.get("choice"), str):
                raise RuntimeError(f"Laya response is missing a valid {field} choice")
            decision[field] = answer["choice"]
            probs = answer.get("probabilities", {})
            chosen_prob = probs.get(answer["choice"]) if isinstance(probs, dict) else None
            confidence = answer.get("answer_confidence", chosen_prob)
            if confidence is None:
                confidence = answer.get("confidence", 0.0)
            try:
                decision[f"{field}_confidence"] = float(confidence)
            except (TypeError, ValueError):
                decision[f"{field}_confidence"] = 0.0

        if decision["severity"] not in {"normal", "warning", "critical"}:
            raise RuntimeError(f"Laya returned unknown severity {decision['severity']!r}")
        if decision["action"] not in self.ACTIONS:
            raise RuntimeError(f"Laya returned unknown action {decision['action']!r}")
        if decision["reward_profile"] not in self.REWARD_PROFILES:
            raise RuntimeError(
                f"Laya returned unknown reward profile {decision['reward_profile']!r}"
            )

        self._cached_state = key
        self._cached_decision = decision
        return decision

    def _rule_severity(self, state: np.ndarray) -> tuple[str, str]:
        values, _ = self._state_key(state)
        omega = float(values[0])
        temperature = float(values[3])
        delta = float(values[5])
        if delta > self.delta_crit - 0.1:
            return "critical", f"Degradation {delta:.3f} is nearing limit {self.delta_crit:.3f}."
        if temperature > 0.95 * self.T_max:
            return "critical", f"Bearing temperature {temperature:.1f} C is nearing limit."
        if temperature > 0.85 * self.T_max:
            return "warning", f"Bearing temperature {temperature:.1f} C is elevated."
        if omega > self.omega_max or omega < self.omega_min:
            return "warning", "Rotor speed is outside its configured operating range."
        return "normal", "No configured hard safety warning is active."

    def assess(self, state: np.ndarray) -> tuple[str, str]:
        """Return none/warning/critical; hard-limit checks override Laya."""
        decision = self._get_decision(state)
        rule_severity, reason = self._rule_severity(state)
        if rule_severity == "critical":
            return "critical", f"CRITICAL: {reason}"

        model_level = decision["severity"]
        if model_level == "critical":
            return "critical", (
                "LAYA classified the observation as critical "
                f"(answer confidence {decision['severity_confidence']:.3f})."
            )
        if model_level == "warning" and rule_severity == "normal":
            return "warning", (
                "LAYA classified the observation as a warning "
                f"(answer confidence {decision['severity_confidence']:.3f})."
            )
        if rule_severity == "warning":
            return "warning", f"WARNING: {reason}"
        return "none", (
            "Normal operation (no rule warning; Laya classified normal "
            f"with answer confidence {decision['severity_confidence']:.3f})."
        )

    def detect_anomaly(self, state: np.ndarray) -> tuple[bool, str]:
        severity, description = self.assess(state)
        return severity == "critical", description

    def act(self, state: np.ndarray) -> np.ndarray:
        """Return Laya's argmax action template, clipped to the env contract."""
        decision = self._get_decision(state)
        action = self.ACTIONS[decision["action"]]
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    def get_reward_weights(self, state: np.ndarray) -> tuple[dict[str, float], float]:
        """Return Laya's selected fixed reward profile and answer confidence."""
        decision = self._get_decision(state)
        profile = decision["reward_profile"]
        confidence = decision["reward_profile_confidence"]
        return dict(self.REWARD_PROFILES[profile]), confidence

    def embed(self, state: np.ndarray) -> torch.Tensor:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.embed_net(state_t)

    def get_embedding(self, state: torch.Tensor) -> torch.Tensor:
        return self.embed_net(state)

    def update_context(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        """Retain the legacy bounded history interface for other components."""
        self.context_history.append({
            "state": np.asarray(state).tolist(),
            "action": np.asarray(action).tolist(),
            "reward": float(reward),
        })
        if len(self.context_history) > self.max_context:
            self.context_history.pop(0)

    def health_check(self) -> bool:
        """Verify that the configured Laya HTTP service is reachable."""
        from urllib.parse import urlsplit, urlunsplit

        parsed = urlsplit(self.endpoint)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for route in ("/health", "/healthz"):
            url = urlunsplit((parsed.scheme, parsed.netloc, route, "", ""))
            response = self._session.get(url, headers=headers, timeout=self.timeout_s)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            return True
        raise RuntimeError("Laya service has neither /health nor /healthz")