"""
Agent 3: Foundation Model (LLM) Agent.

High-level reasoning, reward weight suggestion, anomaly detection.
Provides:
  - Natural language reasoning about turbine state
  - Suggested reward weights w_FM for dynamic shaper
  - Anomaly flags + recommended actions
  - Semantic embedding z_FM for shared space
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict


class FMAgent:
    """
    Foundation Model agent using LLM for high-level reasoning.
    Falls back to rule-based heuristics when no API is available.
    """

    def __init__(
        self,
        state_dim: int = 8,
        model_name: str = "gpt-oss-120b",
        temperature: float = 0.3,
        confidence_threshold: float = 0.7,
        device: str = "cpu",
        use_api: bool = False,
        api_key: str = None,
        api_timeout_s: float = 8.0,
        api_interval: int = 50,
        api_base_url: str = None,
        omega_min: float = 0.5,
        omega_max: float = 1.5,
        T_max: float = 100.0,
        delta_crit: float = 0.9,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.conf_threshold = confidence_threshold
        self.device = device
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.T_max = T_max
        self.delta_crit = delta_crit
        self.api_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        self.api_base_url = (
            api_base_url or os.environ.get("OPENAI_API_BASE", "https://api.siemens.com/llm/v1")
        ).rstrip("/")
        self.use_api = bool(use_api and self.api_key)
        self.api_timeout_s = float(api_timeout_s)
        self.api_interval = max(1, int(api_interval))
        self._api_step_counter = 0
        self._cached_api_weights = None
        self._cached_api_confidence = None
        self._api_executor = ThreadPoolExecutor(max_workers=1) if self.use_api else None
        self._api_future = None
        if use_api and not self.api_key:
            print("[FM] use_api=True but no API key found. Falling back to heuristics.")
        elif self.use_api:
            print(f"[FM] API mode enabled with model={self.model_name} (interval={self.api_interval} steps)")

        # Semantic embedding network (maps state → FM-like embedding)
        self.embed_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
        ).to(device)

        # Context history for in-context learning
        self.context_history = []
        self.max_context = 10

    def __del__(self):
        if getattr(self, "_api_executor", None) is not None:
            self._api_executor.shutdown(wait=False, cancel_futures=True)

    def _submit_api_request(self, state: np.ndarray):
        if not self.use_api or self._api_executor is None:
            return
        if self._api_future is None:
            state_copy = np.array(state, dtype=np.float32, copy=True)
            self._api_future = self._api_executor.submit(self._query_api_reward_weights, state_copy)

    def _poll_api_result(self):
        if self._api_future is None:
            return
        if not self._api_future.done():
            return
        try:
            result = self._api_future.result()
            if result is not None:
                self._cached_api_weights, self._cached_api_confidence = result
        except Exception:
            pass
        finally:
            self._api_future = None

    def _query_api_reward_weights(self, state: np.ndarray):
        """Query external FM API for reward weights and confidence."""
        labels = ["omega", "v_wind", "T_gen", "T_bearing", "P_out", "delta", "grid_demand", "weather"]
        state_dict = {lbl: float(state[i]) for i, lbl in enumerate(labels) if i < len(state)}
        prompt = (
            "Given turbine state JSON, return ONLY JSON with keys: "
            "energy, rul, safety, physics, curiosity, confidence. "
            "All weights in [0,1], confidence in [0,1], and weights sum to 1.\n"
            f"state={json.dumps(state_dict)}"
        )
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "You are a wind turbine control advisor."},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            endpoint = f"{self.api_base_url}/chat/completions"
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.api_timeout_s)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = self._parse_json_payload(content)
            if not parsed:
                return None
            keys = ["energy", "rul", "safety", "physics", "curiosity"]
            if not all(k in parsed for k in keys):
                return None
            weights = {k: float(parsed[k]) for k in keys}
            confidence = float(parsed.get("confidence", 0.6))
            weights = {k: float(np.clip(v, 0.0, 1.0)) for k, v in weights.items()}
            total = sum(weights.values())
            if total <= 1e-6:
                return None
            weights = {k: v / total for k, v in weights.items()}
            confidence = float(np.clip(confidence, 0.0, 1.0))
            print(f"[FM] API response received – weights={weights}, confidence={confidence:.3f}")
            return weights, confidence
        except Exception as exc:
            print(f"[FM] API request failed: {exc}")
            return None

    def _parse_json_payload(self, content: str):
        if not content:
            return None
        text = content.strip()
        if "```" in text:
            lines = text.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except Exception:
            return None

    def get_weight_adjustment(self, z_task, regime_id: int, context_encoder=None):
        """Return regime-based reward weight adjustments."""
        if regime_id is not None:
            regime_id = int(regime_id)
        if regime_id == 0:
            adj = {"energy": -0.03, "rul": 0.02, "safety": 0.05, "physics": 0.03, "curiosity": -0.02, "degradation": -0.05}
        elif regime_id == 1:
            adj = {"energy": -0.05, "rul": 0.05, "safety": 0.08, "physics": 0.04, "curiosity": -0.02, "degradation": -0.10}
        elif regime_id == 2:
            adj = {"energy": -0.08, "rul": 0.08, "safety": 0.12, "physics": 0.05, "curiosity": -0.03, "degradation": -0.14}
        else:
            adj = {"energy": -0.10, "rul": 0.12, "safety": 0.18, "physics": 0.05, "curiosity": -0.05, "degradation": -0.20}
        confidence = 0.6
        if context_encoder is not None and z_task is not None and hasattr(context_encoder, "regime_centroids") and regime_id in context_encoder.regime_centroids:
            try:
                centroid = context_encoder.regime_centroids[regime_id]
                sim = torch.nn.functional.cosine_similarity(z_task.detach().unsqueeze(0), centroid.unsqueeze(0)).item()
                confidence = float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
            except Exception:
                pass
        scaled = {k: v * confidence for k, v in adj.items()}
        return scaled, confidence
        """
        High-level action suggestion.
        Uses rule-based heuristics (FM reasoning simulated).
        """
        omega = state[0] if len(state) > 0 else 1.0
        v_wind = state[1] if len(state) > 1 else 10.0
        T_bearing = state[3] if len(state) > 3 else 50.0
        delta = state[5] if len(state) > 5 else 0.0

        # FM-style reasoning heuristics
        pitch = 0.0
        torque = 0.0
        maint = -1.0

        if delta > 0.7:
            # Critical degradation: reduce load, flag maintenance
            pitch = 0.5
            torque = -0.3
            maint = 1.0
        elif T_bearing > 80.0:
            # High temperature: reduce torque
            pitch = 0.3
            torque = -0.5
        elif v_wind > 20.0:
            # High wind: increase pitch for protection
            pitch = 0.7

        return np.array([pitch, torque, maint], dtype=np.float32)

    def get_reward_weights(self, state: np.ndarray) -> tuple:
        """
        Suggest reward weights based on current state context.
        Uses async API when available, falls back to heuristics.
        Returns (weights_dict, confidence).
        """
        self._api_step_counter += 1
        self._poll_api_result()
        should_query = (
            self._cached_api_weights is None
            or (self._api_step_counter % self.api_interval == 0)
        )
        if should_query:
            self._submit_api_request(state)
        if self._cached_api_weights is not None:
            return self._cached_api_weights, self._cached_api_confidence

        omega = state[0] if len(state) > 0 else 1.0
        T_bearing = state[3] if len(state) > 3 else 50.0
        delta = state[5] if len(state) > 5 else 0.0
        grid_demand = state[6] if len(state) > 6 else 0.5

        confidence = 0.5
        if delta > 0.7:
            weights = {"energy": 0.10, "rul": 0.60, "safety": 0.20, "physics": 0.05, "curiosity": 0.05}
            confidence = 0.9
        elif T_bearing > 85.0:
            weights = {"energy": 0.10, "rul": 0.20, "safety": 0.50, "physics": 0.15, "curiosity": 0.05}
            confidence = 0.85
        elif grid_demand > 0.8:
            weights = {"energy": 0.50, "rul": 0.20, "safety": 0.15, "physics": 0.10, "curiosity": 0.05}
            confidence = 0.75
        else:
            weights = {"energy": 0.30, "rul": 0.25, "safety": 0.20, "physics": 0.15, "curiosity": 0.10}
            confidence = 0.6
        return weights, confidence

    def act(self, state: np.ndarray) -> np.ndarray:
        """High-level action suggestion using rule-based heuristics."""
        omega = state[0] if len(state) > 0 else 1.0
        v_wind = state[1] if len(state) > 1 else 10.0
        T_bearing = state[3] if len(state) > 3 else 50.0
        delta = state[5] if len(state) > 5 else 0.0
        pitch = 0.0; torque = 0.0; maint = -1.0
        if delta > 0.7:
            pitch = 0.5; torque = -0.3; maint = 1.0
        elif T_bearing > 80.0:
            pitch = 0.3; torque = -0.5
        elif v_wind > 20.0:
            pitch = 0.7
        return np.array([pitch, torque, maint], dtype=np.float32)

    def assess(self, state: np.ndarray) -> tuple:
        """
        Grade the current state. Returns (severity, description) where
        severity is one of "none", "warning", "critical".

        Only "critical" means failure is imminent and the arbiter should hand
        full control to the physics safety controller. Thresholds are derived
        from the configured operating constraints rather than hardcoded.
        """
        omega = float(state[0]) if len(state) > 0 else 1.0
        T_bearing = float(state[3]) if len(state) > 3 else 50.0
        delta = float(state[5]) if len(state) > 5 else 0.0

        if delta > self.delta_crit - 0.1:
            return "critical", f"CRITICAL: Degradation index {delta:.2f} nearing limit {self.delta_crit}."
        if T_bearing > 0.95 * self.T_max:
            return "critical", f"CRITICAL: Bearing temperature {T_bearing:.1f} nearing limit {self.T_max}."
        if T_bearing > 0.85 * self.T_max:
            return "warning", f"WARNING: Bearing temperature {T_bearing:.1f} elevated."
        if omega > self.omega_max or omega < self.omega_min:
            return "warning", "WARNING: Rotor speed outside normal operating range."
        return "none", "Normal operation."

    def detect_anomaly(self, state: np.ndarray) -> tuple:
        """
        Emergency flag: True only for critical conditions.

        Rotor-speed excursions are warnings, not emergencies: the hard omega
        bounds are already enforced on every action by the PINN safe-action
        projection, so they do not additionally warrant handing full control
        to the safety controller.
        """
        severity, description = self.assess(state)
        return severity == "critical", description

    def embed(self, state: np.ndarray) -> torch.Tensor:
        """Compute semantic embedding for shared space."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.embed_net(state_t)
        return z

    def get_embedding(self, state: torch.Tensor) -> torch.Tensor:
        """Get embedding for shared space (batched)."""
        return self.embed_net(state)

    def build_prompt(self, state: np.ndarray) -> str:
        """Build structured prompt for LLM reasoning."""
        labels = ["omega", "v_wind", "T_gen", "T_bearing", "P_out", "delta", "grid_demand", "weather"]
        state_desc = "\n".join(f"  {l}: {state[i]:.3f}" for i, l in enumerate(labels) if i < len(state))

        prompt = f"""System: You are an expert wind turbine control advisor.
Current State:
{state_desc}

Analyze the turbine state and recommend:
1. Control action priority (energy vs safety vs RUL)
2. Suggested reward weights (energy, rul, safety, physics, curiosity) summing to 1.0
3. Any anomaly flags

Respond in JSON format."""
        return prompt

    def update_context(self, state: np.ndarray, action: np.ndarray, reward: float):
        """Update context history for in-context learning."""
        self.context_history.append({
            "state": state.tolist(),
            "action": action.tolist(),
            "reward": float(reward),
        })
        if len(self.context_history) > self.max_context:
            self.context_history.pop(0)
