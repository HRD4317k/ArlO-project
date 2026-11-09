"""
HMARL-DRT Main Training Loop.

End-to-end training pipeline implementing the 10-step per-timestep workflow:
  1. Observe state
  2. Encode into shared embedding
  3. Propose actions from all agents
  4. Arbiter gating + safe projection
  5. Execute in environment
  6. Compute dynamic reward
  7. Store transition
  8. Update individual agents
  9. Cross-transfer knowledge
  10. Log and evaluate
"""

import os
import sys
import signal
import torch
import numpy as np
import yaml

# Ensure the package root is on the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agents.sac_agent import SACAgent
from agents.pinn_agent import PINNAgent
from agents.gp_agent import GPAgent
from agents.ipae_agent import IPAEAgent
from agents.fm_agent import FMAgent
from agents.laya_agent import LayaAgent
from core.embedding import SharedEmbeddingSpace
from core.arbiter import Arbiter
from core.reward_shaper import DynamicRewardShaper
from core.transfer_hub import TransferLearningHub
from core.ewc_manager import EWCManager
from core.cemrl_arbiter import CEMRLArbiter
from envs.wind_turbine_env import WindTurbineEnv
from utils.replay_buffer import ReplayBuffer
from utils.logger import Logger
from utils.metrics import MetricsTracker
from utils.regime_detector import RegimeDetector


def build_supervised_z_task(regime_id, delta, n_regimes: int = 4, z_dim: int = 32, device: str = "cpu"):
    """
    Deterministic supervised task embedding — replaces the learned PEARL
    context encoder. Regime detection in this codebase is already fully
    supervised (RegimeDetector thresholds the ground-truth degradation index
    `delta`), so the task embedding fed to the CEMRL arbiter / EWC penalty
    can be built directly from (regime_id, delta) with no learned component:
    a one-hot regime indicator plus the raw delta value in the next slot.

    Returns a 1-D tensor [z_dim] — CEMRLArbiter.select_action unsqueezes it as
    needed, and add_transition stores it for torch.stack into [B, z_dim].
    """
    z = torch.zeros(z_dim, device=device)
    if regime_id is not None and 0 <= regime_id < min(n_regimes, z_dim):
        z[regime_id] = 1.0
    delta_slot = min(n_regimes, z_dim - 1)
    z[delta_slot] = float(delta)
    return z


def safety_urgency(state, omega_min: float, omega_max: float, T_max: float, margin_frac: float = 0.15) -> float:
    """
    Continuous, anticipatory proximity-to-limit signal in [0, 1].

    0 while omega/T_bearing sit within the inner (1 - margin_frac) of their
    legal range; ramps linearly to 1 exactly at the hard boundary. This gives
    the arbiter blend a real-time safety pull that fires BEFORE a limit is
    crossed, rather than only after (which is what a boundary check on the
    resulting state would give).
    """
    omega = float(state[0]) if len(state) > 0 else 1.0
    T_bearing = float(state[3]) if len(state) > 3 else 50.0

    center = 0.5 * (omega_min + omega_max)
    half = 0.5 * (omega_max - omega_min)
    om_prox = abs(omega - center) / half if half > 0 else 0.0
    om_urgency = np.clip((om_prox - (1.0 - margin_frac)) / margin_frac, 0.0, 1.0)

    T_urgency = np.clip((T_bearing - (1.0 - margin_frac) * T_max) / (margin_frac * T_max), 0.0, 1.0)

    return float(max(om_urgency, T_urgency))


def get_device(cfg_device: str = "auto") -> str:
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return cfg_device


def train(config_path: str = "configs/wind_config.yaml"):
    # ── Load Config ──────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "auto"))
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Device: {device} | Seed: {seed}")

    # ── Environment ──────────────────────────────────
    env_cfg = cfg.get("env", {})
    env = WindTurbineEnv(env_cfg)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ── Agent Configuration ──────────────────────────
    agent_cfg = cfg.get("agents", {})
    embed_cfg = cfg.get("embedding", {})
    shared_dim = embed_cfg.get("shared_dim", 64)

    sac_cfg = agent_cfg.get("sac", {})
    nn_agent = SACAgent(
        state_dim=state_dim,
        shared_dim=shared_dim,
        action_dim=action_dim,
        hidden_dim=sac_cfg.get("hidden_dim", 256),
        lr=sac_cfg.get("lr_actor", 3e-4),
        gamma=sac_cfg.get("gamma", 0.99),
        tau=sac_cfg.get("tau", 0.005),
        alpha=sac_cfg.get("alpha", 0.2),
        auto_entropy=sac_cfg.get("auto_entropy", True),
        device=device,
    )

    pinn_cfg = agent_cfg.get("pinn", {})
    physics_cfg = env_cfg.get("physics", {})
    constr_cfg = env_cfg.get("constraints", {})
    physics_agent = PINNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        J=float(physics_cfg.get("J", 1e6)),
        B=float(physics_cfg.get("B", 100.0)),
        R=float(physics_cfg.get("R", 40.0)),
        omega_min=float(constr_cfg.get("omega_min", 0.5)),
        omega_max=float(constr_cfg.get("omega_max", 1.5)),
        T_max=float(constr_cfg.get("T_max", 100.0)),
        hidden_dim=int(pinn_cfg.get("hidden_dim", 128)),
        lr=float(pinn_cfg.get("lr", 1e-3)),
        device=device,
    )

    gp_cfg = agent_cfg.get("gp", {})
    gp_backend = gp_cfg.get("backend", "ensemble")
    if gp_backend == "ipae":
        ipae_cfg = cfg.get("ipae", {})
        gp_agent = IPAEAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=ipae_cfg.get("latent_dim", 8),
            window_size=ipae_cfg.get("window_size", 20),
            encoder_hidden=ipae_cfg.get("encoder_hidden", 64),
            lstm_hidden=ipae_cfg.get("lstm_hidden", 64),
            lstm_layers=ipae_cfg.get("lstm_layers", 1),
            rul_hidden=ipae_cfg.get("rul_hidden", [64, 32, 16]),
            mc_dropout=ipae_cfg.get("mc_dropout", 0.1),
            lambda_kl=ipae_cfg.get("lambda_kl", 0.01),
            lambda_rul=ipae_cfg.get("lambda_rul", 1.0),
            lambda_mono=ipae_cfg.get("lambda_mono", 0.5),
            lambda_physics=ipae_cfg.get("lambda_physics", 1.0),
            n_mc_samples=ipae_cfg.get("n_mc_samples", 10),
            lr=gp_cfg.get("lr", 1e-3),
            device=device,
        )
        print("[Agents] GP slot backend: iPAE (physics-informed autoencoder + supervised RUL)")
    else:
        gp_agent = GPAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=gp_cfg.get("lr", 0.01),
            device=device,
        )
        print("[Agents] GP slot backend: ensemble (classic GPAgent)")

    fm_cfg = agent_cfg.get("fm", {})
    fm_backend = str(fm_cfg.get("backend", "legacy")).lower()
    if fm_backend == "laya":
        fm_agent = LayaAgent(
            state_dim=state_dim,
            device=device,
            endpoint=fm_cfg.get("endpoint"),
            model_name=fm_cfg.get("model_name", "english"),
            timeout_s=fm_cfg.get("api_timeout_s", 10.0),
            omega_min=float(constr_cfg.get("omega_min", 0.5)),
            omega_max=float(constr_cfg.get("omega_max", 1.5)),
            T_max=float(constr_cfg.get("T_max", 100.0)),
            delta_crit=float(constr_cfg.get("delta_crit", 0.9)),
        )
        if fm_cfg.get("health_check", True):
            fm_agent.health_check()
        print(f"[Agents] FM slot backend: Laya System-One ({fm_agent.endpoint})")
    elif fm_backend == "legacy":
        fm_agent = FMAgent(
            state_dim=state_dim,
            device=device,
            model_name=fm_cfg.get("model_name", "gpt-oss-120b"),
            use_api=fm_cfg.get("use_api", False),
            api_interval=fm_cfg.get("api_interval", 50),
            confidence_threshold=fm_cfg.get("confidence_threshold", 0.7),
            temperature=fm_cfg.get("temperature", 0.3),
            api_timeout_s=fm_cfg.get("api_timeout_s", 10.0),
            omega_min=float(constr_cfg.get("omega_min", 0.5)),
            omega_max=float(constr_cfg.get("omega_max", 1.5)),
            T_max=float(constr_cfg.get("T_max", 100.0)),
            delta_crit=float(constr_cfg.get("delta_crit", 0.9)),
        )
    else:
        raise ValueError(f"Unsupported FM agent backend: {fm_backend!r}")
    fm_reward_confidence_threshold = float(fm_cfg.get("reward_confidence_threshold", 0.7))

    # ── Core Modules ─────────────────────────────────
    # Agent embedding dims: [NN_encoder_out, GP_encoder_out, FM_embed_out, PINN_encoder_out]
    nn_enc_dim = sac_cfg.get("hidden_dim", 256) // 2   # 128
    gp_enc_dim = 32
    fm_enc_dim = 128
    pinn_enc_dim = pinn_cfg.get("hidden_dim", 128) // 2  # 64

    embedding = SharedEmbeddingSpace(
        agent_dims=[nn_enc_dim, gp_enc_dim, fm_enc_dim, pinn_enc_dim],
        shared_dim=shared_dim,
        n_agents=4,
        lambda_orth=embed_cfg.get("lambda_orth", 0.01),
    ).to(device)
    embed_opt = torch.optim.Adam(embedding.parameters(), lr=1e-3)

    arb_cfg = cfg.get("arbiter", {})
    # Strength of the pull toward the physics agent on sub-critical warnings.
    warning_caution = float(arb_cfg.get("warning_caution", 0.5))
    # Anticipatory safety margin: urgency ramps from 0 to 1 over the outer
    # `safety_margin_frac` of the omega/T_bearing operating range, reaching 1
    # exactly at the hard constraint boundary. This gives the arbiter a
    # real-time, physically-grounded safety prior in addition to whatever
    # proximity-awareness the learned gating network has picked up on its
    # own — CEMRLArbiter's gating network receives raw state (incl. omega,
    # T_bearing) but has no deterministic bias term, unlike the classic
    # Arbiter, so early in training it has no protection against a step that
    # is about to cross a limit.
    safety_margin_frac = float(arb_cfg.get("safety_margin_frac", 0.15))
    arbiter = Arbiter(
        state_dim=state_dim,
        n_agents=4,
        hidden_dim=arb_cfg.get("hidden_dim", 128),
        temperature=arb_cfg.get("temperature", 1.0),
        lr=arb_cfg.get("lr", 1e-3),
        device=device,
    )

    # ── CEMRL Components ─────────────────────────────
    cemrl_cfg = cfg.get("cemrl", {})
    use_cemrl = cemrl_cfg.get("enabled", True)

    ewc_manager = None
    cemrl_arbiter = None
    regime_detector = None
    z_task_dim = 32

    if use_cemrl:
        # EWC Manager for catastrophic forgetting prevention
        ewc_cfg = cemrl_cfg.get("ewc", {})
        ewc_manager = EWCManager(
            lambda_base=float(ewc_cfg.get("lambda_base", 400.0)),
            beta_lambda=float(ewc_cfg.get("beta_lambda", 0.1)),
            gamma_online=float(ewc_cfg.get("gamma_decay", 0.95)),  # gamma_online in actual class
            fisher_clip=float(ewc_cfg.get("fisher_clip", 1e4)),
            device=device,
        )

        # Supervised task embedding dim (replaces PEARL's learned z_task).
        # RegimeDetector already supervises regime identification directly
        # from the ground-truth degradation index, so no learned context
        # encoder is needed here — see build_supervised_z_task().
        z_task_dim = cemrl_cfg.get("z_task_dim", 32)

        # CEMRL Arbiter for meta-learning gating adaptation
        cemrl_arb_cfg = cemrl_cfg.get("cemrl_arbiter", {})
        cemrl_arbiter = CEMRLArbiter(
            state_dim=state_dim,
            z_task_dim=z_task_dim,
            n_agents=4,
            hidden_dim=cemrl_arb_cfg.get("hidden_dim", 128),
            lr_inner=cemrl_arb_cfg.get("alpha_inner", 0.01),
            lr_outer=cemrl_arb_cfg.get("alpha_outer", 0.001),
            kl_delta=cemrl_arb_cfg.get("delta_kl", 0.01),
            n_inner_steps=cemrl_arb_cfg.get("n_inner_steps", 1),
            meta_update_freq=cemrl_arb_cfg.get("meta_update_freq", 100),
            entropy_coef=cemrl_arb_cfg.get("entropy_coef", 0.01),
            device=device,
        )

        # Regime Detector for transition detection
        regime_cfg = cemrl_cfg.get("regime_detector", {})
        regime_detector = RegimeDetector(
            n_hysteresis=regime_cfg.get("hysteresis_buffer", 10),
            dwell_steps=regime_cfg.get("dwell_steps", 0),
        )

        # Link EWC Manager to agents
        nn_agent.ewc_manager = ewc_manager

        print(f"[CEMRL] Enabled: EWC (λ={ewc_cfg.get('lambda_base', 400.0)}, "
              f"fisher_clip={ewc_cfg.get('fisher_clip', 1e4)}), "
              f"Supervised z_task (dim={z_task_dim}, no PEARL), "
              f"MAML (α_inner={cemrl_arb_cfg.get('alpha_inner', 0.01)})")
    else:
        print("[CEMRL] Disabled - using standard HMARL-DRT")

    rew_cfg = cfg.get("reward", {})
    shaper = DynamicRewardShaper(
        initial_weights=rew_cfg.get("initial_weights"),
        kappa=rew_cfg.get("kappa", 0.1),
        lambda_entropy=rew_cfg.get("lambda_entropy", 0.01),
        alpha_rul=rew_cfg.get("alpha_rul", 0.01),
        phase_mid_step=rew_cfg.get("phase_mid_step", 50000),
        phase_late_step=rew_cfg.get("phase_late_step", 150000),
    )

    trans_cfg = cfg.get("transfer", {})
    transfer = TransferLearningHub(
        lambda_pn=trans_cfg.get("lambda_pn", 0.1),
        lambda_ng=trans_cfg.get("lambda_ng", 0.05),
        lambda_gn=trans_cfg.get("lambda_gn", 0.05),
        lambda_np=trans_cfg.get("lambda_np", 0.1),
        lambda_fm=trans_cfg.get("lambda_fm", 0.02),
        fm_conf_threshold=trans_cfg.get("fm_confidence_threshold", 0.7),
    )

    train_cfg = cfg.get("training", {})
    # Only the sequence-based iPAE backend needs stored history windows.
    buffer_window = gp_agent.window_size if gp_backend == "ipae" else 0
    buffer = ReplayBuffer(train_cfg.get("buffer_size", 1000000), window_size=buffer_window)

    def _gp_embed(states, window=None):
        if gp_backend == "ipae":
            return gp_agent.get_embedding(states, window=window)
        return gp_agent.get_embedding(states)

    def _gp_uncertainty(states, window=None):
        if gp_backend == "ipae":
            return gp_agent.get_uncertainty(states, window=window)
        return gp_agent.get_uncertainty(states)

    log_cfg = cfg.get("logging", {})
    # Allow env variable override for log directory (used by backend job manager)
    log_dir = os.environ.get("HMARL_LOG_DIR", log_cfg.get("log_dir", "runs/"))
    logger = Logger(
        log_dir=log_dir,
        use_tensorboard=log_cfg.get("use_tensorboard", True),
    )
    metrics = MetricsTracker()

    # ── Warm-start: optionally load a pre-trained checkpoint ──────────
    warm_start_path = train_cfg.get("warm_start_checkpoint", None)
    if warm_start_path and os.path.exists(warm_start_path):
        print(f"\n[WarmStart] Loading checkpoint: {warm_start_path}")
        ws_ckpt = torch.load(warm_start_path, map_location=device)
        _ws_loaded = []
        _ws_comps = {
            "nn_agent_actor":     (nn_agent.actor,     "sac_actor"),
            "nn_agent_critic":    (nn_agent.critic,    "sac_critic"),
            "gp_agent":           (gp_agent.network,   "gp_network"),
            "pinn_agent":         (physics_agent.pinn, "pinn"),
            "pinn_policy":        (physics_agent.policy, "pinn_policy"),
            "embedding":          (embedding,          "embedding"),
            "arbiter.gating":     (arbiter.gating,     "arbiter"),
        }
        for name, (obj, key) in _ws_comps.items():
            if key in ws_ckpt and hasattr(obj, "load_state_dict"):
                try:
                    obj.load_state_dict(ws_ckpt[key], strict=False)
                    _ws_loaded.append(name)
                except Exception as exc:
                    print(f"  [WarmStart] Skipping {name}: {exc}")
        if cemrl_arbiter is not None and "cemrl_arbiter" in ws_ckpt:
            try:
                cemrl_arbiter.gating.load_state_dict(ws_ckpt["cemrl_arbiter"], strict=False)
                _ws_loaded.append("cemrl_arbiter.gating")
            except Exception as exc:
                print(f"  [WarmStart] Skip cemrl_arbiter.gating: {exc}")
        # No PEARL context encoder to warm-start (removed in favor of a
        # deterministic supervised z_task built from regime_id + true delta).
        # Reset EWC Fisher from checkpoint — loaded values are stale/huge
        if ewc_manager is not None:
            ewc_manager.fisher = {}
            ewc_manager.optimal_params = {}
        print(f"[WarmStart] Loaded: {', '.join(_ws_loaded) if _ws_loaded else 'nothing'}\n")
    elif warm_start_path:
        print(f"[WarmStart] Path not found, starting from scratch: {warm_start_path}")

    # ── Training Loop ────────────────────────────────
    n_episodes = train_cfg.get("n_episodes", 1000)
    batch_size = train_cfg.get("batch_size", 256)
    update_freq = train_cfg.get("update_freq", 4)
    warmup_steps = train_cfg.get("warmup_steps", 1000)
    total_steps = 0

    # CEMRL: Tracking variables
    current_regime_id = 0
    regime_buffer = []  # Store transitions for FIM computation
    z_task_current = None  # Deterministic supervised task embedding (no PEARL)

    def build_checkpoint():
        ckpt = {
            "sac_actor": nn_agent.actor.state_dict(),
            "sac_critic": nn_agent.critic.state_dict(),
            "pinn": physics_agent.pinn.state_dict(),
            "pinn_policy": physics_agent.policy.state_dict(),
            "gp_network": gp_agent.network.state_dict(),
            "embedding": embedding.state_dict(),
            "arbiter": arbiter.gating.state_dict(),
        }
        if use_cemrl:
            ckpt["ewc_fisher"] = ewc_manager.fisher
            ckpt["ewc_optimal_params"] = ewc_manager.optimal_params
            ckpt["cemrl_arbiter"] = cemrl_arbiter.gating.state_dict()
            ckpt["regime_history"] = regime_detector.transition_history
        return ckpt

    # Allow stopping the run early (Ctrl-C / SIGTERM) without losing the model:
    # the flag is checked at the episode boundary so a checkpoint is written.
    stop_requested = {"flag": False}

    def _request_stop(signum, frame):
        if not stop_requested["flag"]:
            stop_requested["flag"] = True
            print(f"\n[Shutdown] Signal {signum} received — finishing current episode, then saving.\n")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print(f"\n{'='*60}")
    print(f"  HMARL-DRT Training — Wind Turbine Environment")
    print(f"  Episodes: {n_episodes} | Batch: {batch_size}")
    print(f"  State dim: {state_dim} | Action dim: {action_dim}")
    print(f"  Shared embedding dim: {shared_dim}")
    print(f"{'='*60}\n")

    # Per-episode loss tracking for learning verification
    loss_tracker = {"sac_actor": 0.0, "sac_critic": 0.0, "gp": 0.0, "pinn_policy": 0.0}

    for episode in range(n_episodes):
        # Sample regime for balanced training exposure
        deg_cfg = env_cfg.get("degradation", {})
        if deg_cfg.get("sample_regime_each_episode", False):
            regime_probs = deg_cfg.get("regime_probs", [0.25, 0.25, 0.25, 0.25])
            regime_ranges = deg_cfg.get("regime_delta_ranges", {})
            rid = np.random.choice(len(regime_probs), p=regime_probs)
            delta_range = regime_ranges.get(str(rid), [0.0, 0.25])
            state, _ = env.reset(seed=seed + episode, options={"delta_range": delta_range})
        else:
            state, _ = env.reset(seed=seed + episode)
        if hasattr(gp_agent, "reset_window"):
            gp_agent.reset_window()
        ep_reward = 0.0
        ep_physics_violations = 0
        ep_weight_sum = np.zeros(4, dtype=np.float64)
        ep_emergency_steps = 0
        ep_warning_steps = 0
        ep_regime_steps = np.zeros(4, dtype=np.int64)
        laya_labels = {
            "severity": ("normal", "warning", "critical"),
            "action": ("nominal", "high_wind_pitch", "thermal_derate", "degradation_derate"),
            "reward_profile": ("balanced", "safety_priority", "lifetime_priority", "power_priority"),
        }
        laya_decision_counts = {f"{kind}_{label}": 0 for kind, labels in laya_labels.items() for label in labels}
        laya_confidence_sum = {kind: 0.0 for kind in laya_labels}
        done = False
        ep_steps = 0

        while not done:
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)

            # ── STEP 2: Agent Embeddings → Shared Space ──
            with torch.no_grad():
                z1 = nn_agent.actor.get_embedding(s_t)        # [1, nn_enc_dim]
                z2 = gp_agent.get_embedding(s_t)              # [1, gp_enc_dim]
                z3 = fm_agent.get_embedding(s_t)               # [1, fm_enc_dim]
                z4 = physics_agent.get_embedding(s_t)          # [1, pinn_enc_dim]

                z_shared, alpha = embedding([z1, z2, z3, z4])  # [1, shared_dim]

            # ── STEP 3: Agent Action Proposals ───────────
            proposals = {
                "nn": nn_agent.act(state, z_shared),
                "gp": gp_agent.act(state),
                "fm": fm_agent.act(state),
                "physics": physics_agent.policy_act(state),
            }

            # ── STEP 4: Arbiter Gating + Safe Projection ─
            safe_bounds = physics_agent.get_safe_bounds(state)
            delta = float(state[5]) if len(state) > 5 else 0.0
            uncertainty = gp_agent.get_uncertainty_np(state)
            fm_severity, _ = fm_agent.assess(state)
            is_anomaly = fm_severity == "critical"
            # Blend FM-severity caution with a continuous, anticipatory
            # proximity-to-limit signal so the arbiter pulls toward the
            # physics agent BEFORE a constraint is crossed, not only after.
            urgency = safety_urgency(
                state,
                omega_min=float(constr_cfg.get("omega_min", 0.5)),
                omega_max=float(constr_cfg.get("omega_max", 1.5)),
                T_max=float(constr_cfg.get("T_max", 100.0)),
                margin_frac=safety_margin_frac,
            )
            caution = max(
                warning_caution if fm_severity == "warning" else 0.0,
                urgency,
            )

            # CEMRL: Supervised regime detection + task embedding, computed
            # BEFORE action selection (no PEARL learned encoder, no one-step
            # lag — regime_id/z_task reflect the CURRENT state's true delta).
            if use_cemrl:
                prev_regime = current_regime_id
                _, current_regime_id, transition_confirmed = regime_detector.detect(
                    delta_t=delta,
                    uncertainty=uncertainty,
                    fm_emergency_flag=is_anomaly,
                )
                z_task_current = build_supervised_z_task(
                    current_regime_id, delta, n_regimes=4, z_dim=z_task_dim, device=device
                )

            # Use CEMRL Arbiter if enabled and context is available
            if use_cemrl and cemrl_arbiter is not None and z_task_current is not None:
                action, weights = cemrl_arbiter.select_action(
                    state=state,
                    action_proposals=proposals,
                    safe_bounds=safe_bounds,
                    delta=delta,
                    uncertainty=uncertainty,
                    z_task=z_task_current,
                    emergency=is_anomaly,
                    caution=caution,
                    return_weights=True,
                )
            else:
                action, weights = arbiter.select_action(
                    state, action_proposals=proposals, safe_bounds=safe_bounds,
                    delta=delta, uncertainty=uncertainty,
                    emergency=is_anomaly,
                    caution=caution,
                    return_weights=True,
                )

            ep_weight_sum += np.asarray(weights, dtype=np.float64)
            ep_emergency_steps += int(is_anomaly)
            ep_warning_steps += int(fm_severity == "warning")
            if use_cemrl and 0 <= current_regime_id < 4:
                ep_regime_steps[current_regime_id] += 1

            # ── STEP 5: Environment Step ─────────────────
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # ── STEP 6: Dynamic Reward ───────────────────
            rul_est = gp_agent.estimate_rul(state)
            phys_res = physics_agent.get_ode_residual_value(
                s_t, torch.FloatTensor(action).unsqueeze(0).to(device)
            ).item()

            # FM anomaly check → update reward weights
            fm_weights, fm_conf = fm_agent.get_reward_weights(state)
            if fm_conf > fm_reward_confidence_threshold:
                shaper.fm_update_weights(fm_weights)
            if fm_backend == "laya":
                laya_decision = fm_agent._cached_decision
                for kind, labels in laya_labels.items():
                    label = laya_decision[kind]
                    if label in labels:
                        laya_decision_counts[f"{kind}_{label}"] += 1
                    laya_confidence_sum[kind] += laya_decision[f"{kind}_confidence"]

            reward, r_components = shaper.compute_reward(
                state, action, next_state,
                rul_estimate=rul_est,
                physics_residual=phys_res,
                uncertainty=uncertainty,
            )

            if info.get("physics_violated", False):
                ep_physics_violations += 1

            # ── STEP 7: Store Transition ─────────────────
            buffer.push(state, action, reward, next_state, done)
            ep_reward += reward
            total_steps += 1
            ep_steps += 1

            # ── CEMRL: Regime buffer bookkeeping ──────────
            # (Regime detection + z_task already computed earlier this step,
            # before action selection — see STEP 4 above.)
            if use_cemrl:
                transition = (state, action, reward, next_state)
                regime_buffer.append(transition)

                # If regime changed, trigger Fisher computation
                if transition_confirmed and len(regime_buffer) > 50:
                    print(f"[CEMRL] Regime transition: {prev_regime} → {current_regime_id} at step {total_steps}")

                    # Compute Fisher for SAC actor
                    nn_agent.trigger_regime_fisher_computation(
                        regime_id=prev_regime,
                        data_buffer=regime_buffer,
                        ewc_manager=ewc_manager,
                        embedding=embedding,
                        gp_agent=gp_agent,
                        fm_agent=fm_agent,
                        physics_agent=physics_agent,
                    )

                    # Clear regime buffer for new regime
                    regime_buffer = []

            # ── STEPS 8-9: Agent Updates + Transfer ──────
            if len(buffer) > batch_size and total_steps % update_freq == 0 and total_steps > warmup_steps:
                batch = buffer.sample(batch_size)
                s_batch = torch.FloatTensor(batch["states"]).to(device)
                a_batch = torch.FloatTensor(batch["actions"]).to(device)
                s_next_batch = torch.FloatTensor(batch["next_states"]).to(device)

                # Real temporal history for the sequence-based iPAE encoder
                w_batch = w_next_batch = None
                if "state_windows" in batch:
                    w_batch = torch.FloatTensor(batch["state_windows"]).to(device)
                    w_next_batch = torch.FloatTensor(batch["next_state_windows"]).to(device)

                # Compute shared embeddings for batch
                with torch.no_grad():
                    z1_b = nn_agent.actor.get_embedding(s_batch)
                    z2_b = _gp_embed(s_batch, w_batch)
                    z3_b = fm_agent.get_embedding(s_batch)
                    z4_b = physics_agent.get_embedding(s_batch)

                z_shared_b, _ = embedding([z1_b, z2_b, z3_b, z4_b])

                with torch.no_grad():
                    z1_n = nn_agent.actor.get_embedding(s_next_batch)
                    z2_n = _gp_embed(s_next_batch, w_next_batch)
                    z3_n = fm_agent.get_embedding(s_next_batch)
                    z4_n = physics_agent.get_embedding(s_next_batch)

                z_shared_next, _ = embedding([z1_n, z2_n, z3_n, z4_n])

                # Transfer: Physics → NN (T1)
                transfer.physics_to_nn(nn_agent, physics_agent, s_batch, a_batch)

                # Transfer: GP → NN (T3: uncertainty bonus)
                transfer.gp_to_nn(nn_agent, gp_agent, s_batch)

                # Update SAC agent (with hard regime-id EWC penalty if CEMRL enabled)
                if use_cemrl:
                    sac_losses = nn_agent.update(batch, z_shared_b, z_shared_next, regime_id=current_regime_id)
                else:
                    sac_losses = nn_agent.update(batch, z_shared_b, z_shared_next)

                # Transfer: NN → Physics (T4: residual correction)
                transfer.nn_to_physics(physics_agent, s_batch, a_batch, s_next_batch)

                # Update GP/iPAE agent with batch
                if gp_backend == "ipae":
                    gp_loss = gp_agent.update(
                        s_batch, a_batch, torch.FloatTensor(batch["rewards"]).to(device),
                        window=w_batch, next_window=w_next_batch,
                    )
                else:
                    gp_loss = gp_agent.update(s_batch, a_batch, torch.FloatTensor(batch["rewards"]).to(device))

                # Update PINN agent (dynamics model + learnable policy)
                physics_agent.update(s_batch, a_batch, s_next_batch)
                pinn_policy_loss = physics_agent.update_policy(s_batch, a_batch, s_next_batch)

                # Update Arbiter
                deltas_b = s_batch[:, 5:6] if s_batch.shape[1] > 5 else torch.zeros(batch_size, 1, device=device)
                uncerts_b = _gp_uncertainty(s_batch, w_batch).unsqueeze(1)
                rewards_b = torch.FloatTensor(batch["rewards"]).to(device)
                arbiter.update(s_batch, deltas_b, uncerts_b, rewards_b)

                # Track per-agent losses for learning verification
                loss_tracker["sac_actor"] = sac_losses.get("actor_loss", 0.0)
                loss_tracker["sac_critic"] = sac_losses.get("critic_loss", 0.0)
                loss_tracker["gp"] = gp_loss
                loss_tracker["pinn_policy"] = pinn_policy_loss

                # Update embedding alignment
                align_loss = embedding.alignment_loss([z1_b.detach(), z2_b.detach(), z3_b.detach(), z4_b.detach()])
                embed_opt.zero_grad()
                align_loss.backward()
                embed_opt.step()

                # CEMRL: Meta-update CEMRL Arbiter (MAML outer loop)
                if use_cemrl and cemrl_arbiter is not None and z_task_current is not None:
                    # Per-step transition for CEMRL arbiter
                    cemrl_arbiter.add_transition(
                        regime_id=current_regime_id,
                        state=state,
                        z_task=z_task_current,
                        delta=delta,
                        uncertainty=uncertainty,
                        action=action,
                        reward=reward,
                        done=done,
                    )
                    cemrl_arbiter.step_timestep()

                    # Perform meta-update if enough data collected
                    if cemrl_arbiter.should_perform_meta_update():
                        task_batches = [
                            (rid, buf)
                            for rid, buf in cemrl_arbiter.task_buffers.items()
                            if len(buf) >= 10
                        ]
                        if task_batches:
                            meta_metrics = cemrl_arbiter.maml_outer_loop(task_batches)
                            logger.log_scalar("cemrl/meta_loss", meta_metrics["meta_loss"], step=total_steps)

            # Update FM context
            fm_agent.update_context(state, action, reward)
            state = next_state

        # ── STEP 10: Logging ─────────────────────────
        metrics.update(
            episode=episode,
            ep_reward=ep_reward,
            rul_error=abs(rul_est - 500.0),  # Placeholder true RUL
            physics_violations=ep_physics_violations,
            total_steps=total_steps,
            agent_weights=weights,
        )

        log_data = {
            "reward": ep_reward,
            "steps": ep_steps,
            "phase": shaper.phase,
            "rul_estimate": rul_est,
            "physics_violations": ep_physics_violations,
            "delta": delta,
            "uncertainty": uncertainty,
            **{f"w_{k}": v for k, v in shaper.get_weights().items()},
            **{f"g_{name}": float(w) for name, w in zip(["nn", "gp", "fm", "physics"], ep_weight_sum / max(ep_steps, 1))},
            **{f"gfinal_{name}": float(w) for name, w in zip(["nn", "gp", "fm", "physics"], weights)},
            "emergency_frac": ep_emergency_steps / max(ep_steps, 1),
            "warning_frac": ep_warning_steps / max(ep_steps, 1),
            **{f"regime_frac_{i}": int(ep_regime_steps[i]) / max(ep_steps, 1) for i in range(4)},
            **{f"r_{k}": v for k, v in r_components.items()},
            "loss_sac_actor": loss_tracker["sac_actor"],
            "loss_sac_critic": loss_tracker["sac_critic"],
            "loss_gp": loss_tracker["gp"],
            "loss_pinn_policy": loss_tracker["pinn_policy"],
        }

        if fm_backend == "laya":
            for kind, labels in laya_labels.items():
                log_data[f"laya/{kind}_confidence_mean"] = laya_confidence_sum[kind] / max(ep_steps, 1)
                for label in labels:
                    log_data[f"laya/{kind}_{label}_frac"] = laya_decision_counts[f"{kind}_{label}"] / max(ep_steps, 1)

        # Add CEMRL-specific logging
        if use_cemrl:
            log_data["cemrl/regime_id"] = current_regime_id
            log_data["cemrl/regime_transitions"] = len(regime_detector.transition_history)
            if z_task_current is not None:
                log_data["cemrl/z_task_norm"] = float(torch.norm(z_task_current).item())

        logger.log_episode(episode, log_data, step=total_steps)

        if episode % cfg.get("logging", {}).get("print_freq", 10) == 0:
            regime_str = f"| Regime: {current_regime_id}" if use_cemrl else ""
            gm = ep_weight_sum / max(ep_steps, 1)
            print(
                f"Ep {episode:4d} | R: {ep_reward:7.2f} | "
                f"Phase: {shaper.phase:5s} | RUL: {rul_est:6.1f} | "
                f"δ: {delta:.3f} | "
                f"ḡ=[{gm[0]:.2f},{gm[1]:.2f},{gm[2]:.2f},{gm[3]:.2f}] | "
                f"Violations: {ep_physics_violations} {regime_str}"
            )

        # ── Periodic checkpoint (needed for long runs / resume / validation) ──
        ckpt_every = train_cfg.get("checkpoint_every", 500)
        if ckpt_every and (episode + 1) % ckpt_every == 0:
            save_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            ckpt = build_checkpoint()
            torch.save(ckpt, os.path.join(save_dir, f"ep_{episode + 1:05d}.pt"))
            torch.save(ckpt, os.path.join(save_dir, "latest.pt"))

        if stop_requested["flag"]:
            print(f"[Shutdown] Stopping after episode {episode}.")
            break

    # ── Save Final Model ─────────────────────────────
    save_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = build_checkpoint()

    torch.save(checkpoint, os.path.join(save_dir, "final_model.pt"))

    logger.close()
    summary = metrics.get_summary()
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Mean Reward (last 100): {summary['mean_reward']:.2f}")
    print(f"  Mean Physics VR (last 100): {summary['mean_physics_vr']:.4f}")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/wind_config.yaml"
    train(config)
