# Laya FM-slot experiment

## What changes

The training configuration can select `agents.fm.backend: laya`. `LayaAgent` sends one typed-decision request per distinct observation to the local Laya System-One service. It asks for (1) normal/warning/critical severity, (2) one action from a fixed catalog mapped to a bounded 3-value environment action, and (3) one reward-weight profile. It retains the existing 128-value embedding interface used by the shared embedding module.

Laya is a typed decision model, not a text generator and not a continuous-control policy. It cannot return arbitrary floating point controls; the six action templates in `agents/laya_agent.py` are the complete action proposal set. Its severity proposal is checked against the existing configured hard safety limits; rule-based critical conditions override it. This first experiment records raw confidence but uses Laya's argmax choices; confidence is not calibrated for this domain. A failed/malformed Laya server response raises an error rather than silently reverting to the legacy FM agent.

## GPU service

This workspace currently sees an NVIDIA Quadro RTX 6000 with 24 GiB VRAM. The configured training environment has PyTorch 2.5.1+cu121 and Transformers 4.57.6; the pinned Laya 0.4.2 package declares minimums of PyTorch 2.0 and Transformers 4.48, so it can use this CUDA stack without replacing those packages. Its server extra adds the HTTP dependencies. The environment's local `fastapi`/`pydantic` shims can shadow installed libraries if the service is started from the repository, so start the service from `$HOME`, not the repo root.

On the NVIDIA host:

1. Run `hmarl_drt/scripts/setup_laya_server.sh`. It installs the pinned upstream `laya[serve]` release into the configured CUDA environment (`LAYA_SERVER_PYTHON`, default `/home/ubuntu/venv/bin/python`). It checks CUDA is still available afterward.
2. Start the service with the command printed by the script. This experiment loads only the English checkpoint, binds to `127.0.0.1:8000`, and preloads one model to avoid consuming VRAM for unused checkpoints.
3. Check `GET http://127.0.0.1:8000/health` before training (the adapter also supports `/healthz` on older releases). If the service is on another host, set `agents.fm.endpoint` or `LAYA_BASE_URL` to a reachable private URL and configure `LAYA_API_KEY` on both sides; keep the service off the public internet.
4. Run training from `hmarl_drt/` with `train.py configs/wind_laya_experiment.yaml`; startup checks the configured endpoint and fails early if it is unavailable. Its logs/checkpoints go under `hmarl_drt/runs/laya_systemone_experiment/`, not the existing fresh-run directory.

The server consumes GPU memory alongside the RL trainer. Check free VRAM before starting a training run; on memory pressure, use CPU inference or a separate GPU rather than relying on CUDA OOM recovery. The host filesystem currently has limited free space, so verify available storage before downloading model weights.

## Experimental caveats

- This is not a drop-in quality-equivalent replacement. Generic Laya checkpoints were trained for typed decisions, not wind-turbine control. Upstream reports near-chance performance on its own generic typed-decision benchmark for the base checkpoint; those numbers do not measure turbine performance. Evaluate the actual turbine observations and actions before interpreting rewards or safety metrics.
- A live zero-shot probe has already shown a severe domain mismatch: across 12 observations spanning degradation 0.10–0.88, bearing temperature 45–99 °C and rotor speed 0.4–1.6 rad/s, the JSON-state requests returned the same `normal` severity, `maintenance_derate` action and `safety_priority` reward profile. A second probe with natural-language telemetry also returned invariant labels. The deterministic checks still caught explicit high-temperature/high-degradation cases, but **the base Laya checkpoint should not be used for a full RL training run as a state-responsive agent**.
- The server also warned that at least one shipped choice-temperature entry was invalid and had to be clamped. The returned probabilities are not calibrated for turbine decisions in any case. Preserve them in logs, but do not claim confidence-based safety without held-out calibration.
- The catalog and reward profiles are discrete. Current-state anomaly/action/reward questions are combined into one forward pass and reused by `assess`, `act`, and `get_reward_weights` for the same observation.
- The initial config applies every selected reward-weight profile (`reward_confidence_threshold: 0.0`) and records raw confidence. Fit/validate thresholds on held-out turbine traces; confidence is not proof of correctness.
- Before a full run, fine-tune Laya on expert- or simulator-labelled turbine observations/actions and a separately labelled anomaly set, then rerun a probe that requires the output labels to vary correctly over the operating envelope. Rule-generated labels can bootstrap it, but would teach the model to imitate those rules rather than demonstrate independent policy quality.
- The training config has `deg_acceleration: 1.5` and the existing temperature degradation default. This experiment changes the FM slot, not those environment dynamics.
- For a credible model study, compare Laya with the legacy heuristic FM and with SAC-only under paired seeds, identical environment configs, and the same action safety projection. Report action-selection latency, abstentions, anomaly precision/recall, energy, degradation, and regime crossings.

## License and source

Laya upstream: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), Apache-2.0. The HMARL adapter communicates with the upstream HTTP API; it does not vendor upstream source code.
