# HMARL-DRT: Heterogeneous Multi-Agent Reinforcement Learning with Dynamic Reward Shaping & Bidirectional Transfer Learning

## Research Documentation & Technical Reference

---

## 1. Executive Summary

**HMARL-DRT** is a novel framework that combines four fundamentally different AI agents — Deep RL (SAC), Bayesian/Statistical (GP/iPAE), Foundation Model (LLM), and Physics-Informed Neural Network (PINN) — into a unified, collaboratively learning system. A meta-controller (Arbiter) dynamically routes decisions to the best-suited agent based on the current state, while a Dynamic Reward Shaper adapts the optimization objective across training phases and degradation stages. A Transfer Learning Hub enables bidirectional knowledge flow between all agents.

**Core Innovation:** _"A heterogeneous multi-agent RL framework where neural, statistical, physics, and foundation model agents collaboratively learn a shared policy through bidirectional transfer learning, guided by a dynamic reward function that adapts to degradation stage, training phase, and semantic context."_

### Industrial Use Cases
| Dimension | Wind Turbine Farm | CNC Machining |
|---|---|---|
| State Space | ℝ⁸ | ℝ¹⁴ |
| Action Space | Continuous ℝ³ | Mixed ℝ³ × {0,1} |
| Physics Model | Aerodynamic ODE | Taylor + Merchant |
| Key Challenge | Multi-turbine coordination | Precision + tool life trade-off |

---

## 2. System Architecture Overview

The system is composed of **7 major subsystems** that interact at every timestep:

1. **Industrial Environment** — Emits sensor states, receives actions
2. **Heterogeneous Agent Pool** — 4 agents with different paradigms
3. **Shared Latent Embedding Space** — Attention-weighted alignment layer
4. **Meta-Controller / Arbiter** — MoE-style gating network
5. **Dynamic Reward Shaper** — Phase-adaptive, FM-steered reward
6. **Transfer Learning Hub** — 5 bidirectional knowledge flows
7. **iPAE Module** — Interpretable Physics-informed Autoencoder (Agent 2 specialization)

> See: **Diagram 1 — Full System Architecture**  
> See: **Diagram 2 — Component Interaction Map**

---

## 3. Agent Specifications

### 3.1 Agent 1 — Deep RL Agent (Soft Actor-Critic)

| Property | Detail |
|---|---|
| **Role** | Primary policy optimizer |
| **Algorithm** | SAC (Soft Actor-Critic) |
| **Input** | Raw state `s_t` + shared embedding `z_shared` |
| **Output** | Continuous action `a_t ∈ ℝᵐ` (pitch angle, torque setpoint, maintenance flag) |
| **Architecture** | Gaussian Actor (256-dim hidden) + Twin-Q Critics |

**Objective (Maximum Entropy RL):**
$$\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_t \gamma^t \left(r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))\right)\right]$$

**Transfer Received:**
- Physics constraints as loss regularization (from PINN)
- Uncertainty maps for exploration bonus (from GP/iPAE)
- Semantic reward signals (from Foundation Model)

### 3.2 Agent 2 — Bayesian Agent (Gaussian Process / iPAE)

| Property | Detail |
|---|---|
| **Role** | Uncertainty quantification + safe exploration guide + RUL estimation |
| **Model** | Gaussian Process over state-action space, or iPAE (Interpretable Physics-informed Autoencoder) |
| **Input** | State history `{s_{t-T}, ..., s_t}` or sensor window `X_t ∈ ℝ^{W×F}` |
| **Output** | Action recommendation `μ_a`, epistemic uncertainty `σ²`, RUL distribution `p(RUL|s_{1:t})` |

**GP Posterior:**
$$\mu_{GP}(s,a|\mathcal{D}_t) = \mu_0 + \mathbf{k}^T(K + \sigma_n^2 I)^{-1}(\mathbf{r} - \mu_0)$$

**iPAE Specialization:**  
When instantiated as iPAE, each latent dimension has physical meaning:

| Dim | Physical Meaning | Constraint | Activation |
|---|---|---|---|
| z₁ | Wear Index | Monotone ↑ | Softplus |
| z₂ | Thermal Drift | Bounded | Tanh × C |
| z₃ | Fatigue Accumulation | Monotone ↑ | Softplus |
| z₄ | Imbalance Severity | Oscillatory | Sin-activated |
| z₅ | Lubrication State | Step-wise ↓ | Sigmoid |
| z₆ | Crack Propagation | Monotone ↑ | Softplus |
| z₇ | Misalignment | Bounded | Tanh × C |
| z₈ | Residual/Noise | None | Linear |

**Transfer Sent:** Uncertainty maps → NN exploration, RUL → reward shaper curiosity term

### 3.3 Agent 3 — Foundation Model Agent (LLM-based)

| Property | Detail |
|---|---|
| **Role** | High-level reasoning, reward hypothesis generation, anomaly interpretation |
| **Input** | Structured text summary of state, maintenance logs, weather forecasts |
| **Output** | NL reasoning, suggested reward weights, anomaly flags, high-level actions |

**Example Prompt Flow:**
```
System: You are an expert wind turbine control advisor.
State: Turbine 3 — Bearing temp: 87°C (↑12°C in 2h),
       Vibration: 4.2 mm/s (threshold: 3.5),
       Wind speed: 14 m/s, Grid demand: HIGH,
       Estimated RUL: 48h

Query: What control action and reward priority do you recommend?

FM Output: "Bearing degradation is accelerating. Recommend
            reducing RPM by 15% to extend RUL. Prioritize
            r_RUL weight (w2 → 0.6). Flag for inspection
            within 24h. Energy output secondary."
```

**Transfer Sent:** Reward weight suggestions → Reward Shaper, anomaly flags → Arbiter override

### 3.4 Agent 4 — Physics / PINN Agent

| Property | Detail |
|---|---|
| **Role** | Physics-grounded safe baseline policy + constraint enforcement |
| **Model** | ODE-based turbine dynamics + Physics-Informed Neural Network |
| **Key Contribution** | Safe action bounds, primary agent in early training, physics violation detection |

**Turbine ODE System:**
$$\frac{d\omega}{dt} = \frac{1}{J}(T_{aero}(\lambda, \beta) - T_{gen} - B\omega)$$
$$\frac{dT}{dt} = \frac{1}{C_{th}}(P_{loss}(\omega, T_{gen}) - h(T - T_{amb}))$$
$$\frac{d\delta}{dt} = f_{deg}(\omega, T, F_{vib})$$

**PINN Loss:**
$$\mathcal{L}_{PINN} = \mathcal{L}_{data} + \lambda_1 \mathcal{L}_{ODE} + \lambda_2 \mathcal{L}_{BC}$$

**Transfer Sent:** Constraint bounds + regularization loss → NN Agent

> See: **Diagram 3 — Agent Interaction & Data Flow**

---

## 4. Shared Latent Embedding Space

Each agent encodes the state into its own latent representation, then an attention-weighted alignment maps all to a unified shared space:

$$z_{shared}^t = \sum_{k=1}^{4} \alpha_k^t \cdot W_k z_k^t$$

**Attention weights:**
$$\alpha_k^t = \frac{\exp(e_k^T z_k^t / \sqrt{d})}{\sum_{j=1}^{4} \exp(e_j^T z_j^t / \sqrt{d})}$$

**Alignment Loss (enforces semantic consistency):**
$$\mathcal{L}_{align} = \sum_{k \neq j} \|W_k z_k^t - W_j z_j^t\|_2^2 + \lambda_{orth} \sum_{k} \|W_k^T W_k - I\|_F^2$$

> See: **Diagram 4 — Shared Embedding Architecture**

---

## 5. Meta-Controller / Arbiter

### Competence Score
$$c_k(s_t) = \eta \cdot \bar{R}_k(s_t) - (1-\eta) \cdot \mathcal{U}_k(s_t)$$

### Gating Network
$$g_k(s_t) = \text{softmax}(G_\Theta^{(k)}(s_t) / \tau)$$

### Composite Action (projected onto safe set)
$$a_t = \Pi_{\mathcal{C}_{safe}}\left(\sum_{k=1}^{4} g_k(s_t) \cdot a_k^t\right)$$

### Degradation-Aware Routing (iPAE δ_t drives routing)

| Degradation Zone | δ_t Range | Dominant Agent | Reward Priority |
|---|---|---|---|
| **Healthy** | < 0.3 | Physics (g=0.50) | Energy ↑, Physics ↑ |
| **Degrading** | 0.3–0.7 | NN (g=0.40) | RUL ↑, balanced |
| **Critical** | > 0.7 | iPAE (g=0.50) | RUL=0.6, Safety ↑ |
| **Emergency** | FM flag or σ² > θ | Physics (g=1.0) | Full override |

> See: **Diagram 5 — Degradation-Aware Routing**

---

## 6. Dynamic Reward Function

### Composite Reward
$$r_t = \mathbf{w}(t)^T \mathbf{r}(s_t, a_t, s_{t+1})$$

| Component | Formula | Source |
|---|---|---|
| Energy | $r_{energy} = P_t / P_{rated}$ | Environment |
| RUL | $r_{RUL} = 1 - e^{-\alpha \cdot RUL_t}$ | iPAE / GP |
| Safety | $r_{safety} = -\mathbb{1}[\text{violated}] - \mu \cdot d(a, \mathcal{C}_{safe})$ | PINN |
| Physics | $r_{physics} = -\|\mathcal{L}_{ODE}\|$ | PINN + iPAE |
| Curiosity | $r_{curiosity} = \beta \cdot \sigma^2_{epistemic}$ | iPAE / GP |

### Phase-Based Weight Evolution

| Phase | Episodes | w_energy | w_RUL | w_safety | w_physics | w_curiosity |
|---|---|---|---|---|---|---|
| Early (data-sparse) | 0–50K | 0.10 | 0.20 | 0.30 | 0.30 | 0.10 |
| Mid (learning) | 50K–150K | 0.30 | 0.30 | 0.20 | 0.10 | 0.10 |
| Late (optimizing) | 150K+ | 0.50 | 0.30 | 0.10 | 0.05 | 0.05 |
| Degradation Event | FM-triggered | 0.10 | **0.60** | 0.20 | 0.05 | 0.05 |

**Weight Dynamics ODE:**
$$\frac{d\mathbf{w}}{dt} = -\nabla_{\mathbf{w}} \mathcal{L}_{weight} + \kappa(\mathbf{w}_t^{FM} - \mathbf{w}(t))$$

---

## 7. Transfer Learning Hub — 5 Bidirectional Paths

| Transfer Flow | Direction | Mechanism | Purpose |
|---|---|---|---|
| **T1** | Physics → NN | ODE residual as regularization loss | Physics consistency |
| **T2** | NN → GP/iPAE | KL divergence distillation | GP tracks NN in high-confidence regions |
| **T3** | GP/iPAE → NN | Uncertainty map as exploration bonus | Efficient exploration |
| **T4** | NN → Physics | Residual correction network | Data-driven parameter adaptation |
| **T5** | FM → All | Reward weights + semantic embeddings + anomaly flags | Semantic steering |

**Unified Transfer Loss:**
$$\mathcal{L}_{transfer} = \mathcal{L}_{P \rightarrow N} + \mathcal{L}_{N \rightarrow G} + \mathcal{L}_{G \rightarrow N} + \mathcal{L}_{N \rightarrow P} + \mathcal{L}_{FM \rightarrow All}$$

> See: **Diagram 6 — Transfer Learning Flows**

---

## 8. iPAE (Interpretable Physics-informed Autoencoder) Deep Dive

### Architecture Stack
1. **Temporal Encoder:** Conv1D → BiLSTM → Attention Pooling → h_pooled ∈ ℝ⁵¹²
2. **Variational Latent:** h → μ_z, log σ²_z → z_t ~ N(μ_z, σ²_z) ∈ ℝ⁸
3. **Physics Constraints:** Monotonicity, boundedness, ODE residual on latent dims
4. **Decoder:** Linear → BiLSTM → ConvTranspose1D → X̂_t
5. **RUL Head:** MLP (64→32→16→1) with MC Dropout for uncertainty

### iPAE Loss Function
$$\mathcal{L}_{iPAE} = \underbrace{\|X_t - \hat{X}_t\|_F^2}_{\text{reconstruction}} + \lambda_p \underbrace{\|dz/dt - f_{ODE}(z)\|^2}_{\text{physics}} + \lambda_r \underbrace{\|\hat{RUL}_t - RUL_t^{true}\|^2}_{\text{RUL}} + \lambda_m \underbrace{\sum_{k \in \{1,3,6\}} \max(0, z_k(t) - z_k(t+1))}_{\text{monotonicity}} + \lambda_{KL} \underbrace{KL(q(z) \| p(z))}_{\text{VAE prior}}$$

### iPAE → HMARL Integration Points
- **z_iPAE → Shared Embedding** (as z₂, projected via W₂)
- **RUL_t → Dynamic Reward Shaper** (r_RUL = 1 - exp(-α · RUL_t))
- **σ²_RUL → Exploration Bonus** (curiosity reward)
- **δ_t → Arbiter Gating** (degradation-aware routing)
- **Physics Residual → Transfer Hub** (PINN parameter correction)

> See: **Diagram 7 — iPAE Internal Architecture**

---

## 9. End-to-End Training Flow

### Per-Timestep Execution (10 Steps)

| Step | Action | Components Involved |
|---|---|---|
| 1 | **Observe** state s_t from environment | Environment |
| 2 | **Encode** into shared latent z_shared | All Encoders + Embedding |
| 3 | **Propose** actions from all 4 agents | Agent Pool |
| 4 | **Gate** via Arbiter + project onto safe set | Arbiter + PINN bounds |
| 5 | **Execute** combined action in environment | Environment |
| 6 | **Compute** dynamic reward (5 components) | Reward Shaper + iPAE + PINN |
| 7 | **Update** individual agents | SAC, GP, FM context, PINN |
| 8 | **Transfer** knowledge bidirectionally | Transfer Hub (5 paths) |
| 9 | **Update** Arbiter gating weights | Arbiter policy gradient |
| 10 | **Log** metrics and evaluate | Logger + Metrics |

> See: **Diagram 8 — End-to-End Training Flow**

---

## 10. Theoretical Guarantees

### Theorem 1: Convergence
Under Lipschitz policies, differentiable gating, and bounded rewards, the composite policy converges to a local Nash equilibrium with rate $O(1/\sqrt{t})$.

### Theorem 2: Safety
With projection $\Pi_{\mathcal{C}_{safe}}$ at every timestep: $\mathbb{P}(a_t \in \mathcal{C}_{safe}(s_t)\ \forall t) = 1$

### Theorem 3: Sample Complexity Reduction
$$N_{HMARL} \leq N_{single} \cdot (1 - \rho_{physics})$$
where $\rho_{physics}$ is the physics informativeness ratio.

### Proposition 1: Reward Non-Collapse
Dynamic weights don't collapse to a single component if $\lambda_{entropy} > \max_k |\partial \mathbb{E}[Q] / \partial w_k|$.

---

## 11. Evaluation & Ablation Design

### Ablation Matrix (8 variants)

| ID | Variant | Removed | Most Affected Metric |
|---|---|---|---|
| A1 | HMARL\P | Physics Agent | Physics Violation Rate |
| A2 | HMARL\B | Bayesian Agent | RUL MAE, Sample Efficiency |
| A3 | HMARL\FM | Foundation Model | Reward Adaptation Speed |
| A4 | HMARL\DR | Dynamic Reward | Cumulative Reward (late) |
| A5 | HMARL\TL | Transfer Learning | Sample Complexity |
| A6 | HMARL\SE | Shared Embedding | Agent Coordination |
| A7 | HMARL\AR | Arbiter (random) | Reward Variance |
| A8 | SAC-Single | Everything | All metrics |

### Key Metrics
| Metric | Symbol | Target |
|---|---|---|
| Cumulative Reward | R_cum | Maximize |
| RUL Mean Abs Error | MAE_RUL | Minimize |
| Physics Violation Rate | VR_physics | → 0% |
| Sample Efficiency | N_sample | Minimize |
| Reward Variance | σ²_R | Minimize |
| Adaptation Time | T_adapt | Minimize |

### Expected Results (HMARL-DRT vs baselines)
| Method | R_cum ↑ | MAE_RUL ↓ | VR_phys ↓ | N_sample ↓ |
|---|---|---|---|---|
| SAC-Single | 1.00 | 48.2h | 12.3% | 100K |
| MBPO | 1.18 | 38.7h | 8.1% | 55K |
| **HMARL-DRT** | **1.38** | **28.4h** | **2.1%** | **42K** |

---

## 12. Project Structure

```
hmarl_drt/
├── agents/
│   ├── __init__.py
│   ├── sac_agent.py          # Agent 1: Deep RL (SAC)
│   ├── gp_agent.py           # Agent 2: Bayesian GP
│   ├── ipae_agent.py         # Agent 2 alt: iPAE
│   ├── fm_agent.py           # Agent 3: Foundation Model
│   └── pinn_agent.py         # Agent 4: Physics PINN
├── core/
│   ├── __init__.py
│   ├── arbiter.py            # Meta-controller / Gating
│   ├── reward_shaper.py      # Dynamic reward function
│   ├── transfer_hub.py       # Transfer learning manager
│   └── embedding.py          # Shared latent space
├── envs/
│   ├── __init__.py
│   ├── wind_turbine_env.py   # Wind turbine Gym environment
│   └── cnc_env.py            # CNC machining Gym environment
├── ipae/
│   ├── __init__.py
│   ├── encoder.py            # Conv1D + BiLSTM encoder
│   ├── decoder.py            # Reconstruction decoder
│   ├── rul_head.py           # RUL estimation + MC Dropout
│   ├── physics_module.py     # ODE constraints on latent
│   └── pretrain.py           # iPAE pre-training script
├── utils/
│   ├── replay_buffer.py
│   ├── metrics.py
│   └── logger.py
├── experiments/
│   ├── ablation_runner.py
│   └── benchmark_runner.py
├── configs/
│   ├── wind_config.yaml
│   └── cnc_config.yaml
├── dashboard/
│   ├── app.py                # Streamlit/Gradio monitoring UI
│   └── components/
├── tests/
│   ├── test_agents.py
│   ├── test_transfer.py
│   └── test_environment.py
├── train.py                  # Main training entry point
├── evaluate.py               # Evaluation & plotting
├── requirements.txt
└── README.md
```

---

_Generated: May 2026 | HMARL-DRT Research Documentation v1.0_
