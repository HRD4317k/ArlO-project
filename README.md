<div align="center">

# ⚡ HMARL-DRT

### Heterogeneous Multi-Agent Reinforcement Learning with Dynamic Reward Shaping & Bidirectional Transfer Learning

*A novel framework for predictive maintenance and adaptive control of industrial systems*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Proprietary-gray.svg)]()

</div>

---

## Research Motivation

Industrial predictive maintenance demands reasoning across multiple paradigms simultaneously — data-driven pattern recognition, physics-based modeling, uncertainty quantification, and high-level strategic planning. No single AI approach excels at all of these.

**HMARL-DRT** addresses this by combining **four fundamentally different AI agents** into a unified, collaboratively learning system:

> *"A heterogeneous multi-agent RL framework where neural, statistical, physics, and foundation model agents collaboratively learn a shared policy through bidirectional transfer learning, guided by a dynamic reward function that adapts to degradation stage, training phase, and semantic context."*

### Key Research Questions

1. **Can heterogeneous agents outperform homogeneous ensembles** by exploiting paradigm-specific strengths at different operational phases?
2. **How should knowledge transfer be structured** between fundamentally different learning paradigms (neural ↔ statistical ↔ physics ↔ semantic)?
3. **Can dynamic reward shaping** guided by degradation state and foundation model context accelerate convergence and improve safety?

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HMARL-DRT Framework                         │
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│  │  Agent 1:    │    │  Agent 2:    │    │  Agent 3:    │            │
│  │  SAC         │    │  GP / iPAE   │    │  Foundation  │            │
│  │  (Deep RL)   │    │  (Bayesian)  │    │  Model       │            │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘            │
│         │                   │                   │                    │
│         │    ┌──────────────┴───────────────┐    │   ┌─────────────┐ │
│         ├────┤  Shared Embedding Space       ├───┤   │  Agent 4:    │ │
│         │    │  (Attention-Weighted Fusion)   │   │   │  PINN        │ │
│         │    └──────────────┬───────────────┘    │   │  (Physics)   │ │
│         │                   │                   │   └──────┬───────┘ │
│         │                   ▼                   │          │         │
│         │         ┌─────────────────┐           │          │         │
│         └────────►│  Arbiter         │◄──────────┘──────────┘         │
│                   │  (MoE Gating)    │                                │
│                   └────────┬────────┘                                │
│                            │ action                                  │
│                            ▼                                         │
│                   ┌─────────────────┐     ┌──────────────────┐      │
│                   │  Wind Turbine    │────►│  Dynamic Reward   │      │
│                   │  Environment     │     │  Shaper           │      │
│                   │  (ODE Dynamics)  │     │  (Phase-Adaptive) │      │
│                   └─────────────────┘     └──────────────────┘      │
│                                                                     │
│                   ┌──────────────────────────────────────┐          │
│                   │  Transfer Learning Hub                │          │
│                   │  (5 Bidirectional Knowledge Flows)    │          │
│                   └──────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

### 7 Core Subsystems

| # | Subsystem | Description |
|---|-----------|-------------|
| 1 | **Wind Turbine Environment** | Realistic ODE-based simulator with 8D state, 3D continuous actions |
| 2 | **Heterogeneous Agent Pool** | SAC (Deep RL) + GP/iPAE (Bayesian) + FM (Foundation) + PINN (Physics) |
| 3 | **Shared Embedding Space** | Attention-weighted fusion aligning agent encodings into $z_{\text{shared}}$ |
| 4 | **Arbiter (Meta-Controller)** | MoE-style gating with degradation-aware routing & emergency overrides |
| 5 | **Dynamic Reward Shaper** | 5-component reward adapting across 3 training phases, steered by FM |
| 6 | **Transfer Learning Hub** | 5 bidirectional knowledge flows: Physics↔NN, NN↔GP, FM→All |
| 7 | **iPAE Module** | Interpretable Physics-informed Autoencoder with 8 semantic latent dims |

---

## Agent Specifications

### Agent 1: SAC (Soft Actor-Critic) — Deep RL
- **Role:** Primary policy optimizer (dominant in mid/late training)
- **Architecture:** Gaussian Actor + Twin-Q Critics (256-dim hidden)
- **Objective:** Maximum entropy RL: $\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_t \gamma^t \left(r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))\right)\right]$
- **Receives:** Physics regularization (PINN), uncertainty bonus (GP), reward weights (FM)

### Agent 2: GP/iPAE — Bayesian Uncertainty
- **Role:** Uncertainty quantification, safe exploration, RUL estimation
- **Architecture:** Deep Ensemble (5 models) with optional iPAE specialization
- **Output:** Mean action $\mu_a$, epistemic uncertainty $\sigma^2$, RUL distribution
- **iPAE Latent Space:** 8 physics-semantic dimensions (wear, thermal drift, fatigue, imbalance, lubrication, crack propagation, misalignment, residual)

### Agent 3: Foundation Model — Strategic Reasoning
- **Role:** High-level guidance, anomaly detection, dynamic reward weight suggestions
- **Architecture:** Rule-based heuristics (LLM-ready interface)
- **Output:** Reward weight modulation, anomaly flags, confidence-gated strategies

### Agent 4: PINN — Physics-Informed Controller
- **Role:** Safety-critical control (dominant in early training & emergencies)
- **Architecture:** ODE-constrained neural network learning turbine dynamics
- **Physics:** Aerodynamic torque $C_p(\lambda, \beta)$, thermal dynamics, degradation accumulation
- **Output:** Conservative actions + dynamic safe bounds

---

## Dynamic Reward Shaping

The reward function adapts across training phases:

$$r_t = \sum_{i=1}^{5} w_i(t) \cdot r_i(s_t, a_t, s_{t+1})$$

| Component | Early Phase | Mid Phase | Late Phase |
|-----------|:-----------:|:---------:|:----------:|
| **Safety** | 0.50 | 0.25 | 0.15 |
| **Physics** | 0.30 | 0.15 | 0.10 |
| **RUL** | 0.20 | 0.35 | 0.25 |
| **Energy** | 0.10 | 0.20 | 0.40 |
| **Curiosity** | 0.10 | 0.05 | 0.10 |

Weight evolution is further influenced by the Foundation Model's suggestions via: $\frac{dw}{dt} += \kappa(w_{\text{FM}} - w)$

---

## Transfer Learning Hub

Five bidirectional knowledge flows between agents:

| Path | Direction | Mechanism | Strength |
|------|-----------|-----------|:--------:|
| T1 | Physics → NN | ODE residual as regularization loss | 0.10 |
| T2 | NN → GP | Actor features inform GP ensemble | 0.05 |
| T3 | GP → NN | Uncertainty bonus for exploration | 0.05 |
| T4 | NN → Physics | Data-driven residual correction | 0.10 |
| T5 | FM → All | Conditional semantic alignment (conf > 0.7) | 0.02 |

---

## Degradation-Aware Routing

The Arbiter dynamically shifts control authority based on component health:

| Phase | δ Range | Dominant Agent | Strategy |
|-------|:-------:|----------------|----------|
| **Healthy** | < 0.3 | PINN (0.50) | Physics-guided conservative control |
| **Degrading** | 0.3–0.7 | SAC (0.50) | Data-driven optimization |
| **Critical** | > 0.7 | GP/iPAE (0.50) | Uncertainty-aware, RUL-focused |
| **Emergency** | Anomaly | PINN (1.00) | Full physics override |

---

## Project Structure

```
hmarl_drt/
├── agents/                 # Heterogeneous agent implementations
│   ├── sac_agent.py       #   Agent 1: Soft Actor-Critic (Deep RL)
│   ├── gp_agent.py        #   Agent 2: GP / iPAE (Bayesian)
│   ├── fm_agent.py        #   Agent 3: Foundation Model (Heuristic-LLM)
│   └── pinn_agent.py      #   Agent 4: Physics-Informed NN
├── core/                   # Core framework modules
│   ├── arbiter.py         #   Meta-controller with MoE gating
│   ├── embedding.py       #   Shared embedding space (attention fusion)
│   ├── reward_shaper.py   #   Dynamic phase-adaptive reward
│   └── transfer_hub.py    #   Bidirectional transfer learning (5 paths)
├── envs/                   # Gymnasium environments
│   └── wind_turbine_env.py #  Wind turbine ODE simulator (8D state, 3D action)
├── ipae/                   # iPAE module (Agent 2 specialization)
│   ├── encoder.py         #   Conv1D + BiLSTM + Attention pooling
│   ├── decoder.py         #   Symmetric reconstruction
│   ├── physics_module.py  #   ODE constraints + monotonicity enforcement
│   ├── rul_head.py        #   RUL estimation with MC dropout
│   └── pretrain.py        #   Pre-training pipeline + synthetic data
├── backend/                # FastAPI REST API
│   ├── main.py            #   App entry point, health, workflow endpoints
│   ├── schemas.py         #   Pydantic request/response models
│   ├── job_manager.py     #   Subprocess-based training orchestration
│   └── routes/            #   API route modules
│       ├── configs.py     #     Config CRUD (list, get, update, clone)
│       ├── training.py    #     Training jobs (start, cancel, status, logs)
│       ├── runs.py        #     Run metrics (browse, compare, download)
│       └── simulation.py  #     Interactive environment rollouts
├── dashboard/              # Streamlit interactive UI
│   ├── app.py             #   Main application with page routing
│   └── components/        #   Modular UI components
│       ├── api_client.py  #     Centralized backend API client
│       ├── charts.py      #     Plotly chart builders (12 chart types)
│       ├── home.py        #     Home page with workflow guide
│       ├── pages.py       #     All page renderers (7 pages)
│       └── settings.py    #     System settings & API reference
├── configs/                # YAML training configurations
│   ├── wind_config.yaml   #   Production config (1000 episodes)
│   └── test_config.yaml   #   Quick test config (50 episodes)
├── utils/                  # Shared utilities
│   ├── logger.py          #   CSV + TensorBoard logging
│   ├── metrics.py         #   7-metric tracking system
│   └── replay_buffer.py   #   Experience replay (1M capacity)
├── runs/                   # Training run outputs (auto-generated)
├── tests/                  # Test suite
├── train.py               # Main training entry point
└── requirements.txt       # Python dependencies
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- pip

### 2. Installation

```bash
# Clone the repository
git clone https://code.siemens.com/predictive-analytics-core/hmarl-drt.git
cd hmarl-drt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r hmarl_drt/requirements.txt
```

### 3. Run Training (CLI)

```bash
cd hmarl_drt

# Quick test run (50 episodes)
python train.py configs/test_config.yaml

# Full training run (1000 episodes)
python train.py configs/wind_config.yaml
```

### 4. Start the Backend API

```bash
cd hmarl_drt
PYTHONPATH=$PWD python -m uvicorn backend.main:app --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 5. Launch the Dashboard

```bash
cd hmarl_drt
PYTHONPATH=$PWD streamlit run dashboard/app.py --server.port 8501
```

Dashboard at: `http://localhost:8501`

---

## User Workflow

The dashboard guides you through a 6-step workflow:

| Step | Action | Description |
|:----:|--------|-------------|
| 1 | **Configure** | Browse/edit training configs — episodes, batch size, reward weights, agent parameters |
| 2 | **Train** | Launch training jobs with chosen config |
| 3 | **Monitor** | Watch real-time reward curves, episode logs, job status |
| 4 | **Analyse** | Deep-dive into completed runs — gating dynamics, degradation, rewards |
| 5 | **Compare** | Side-by-side comparison of training runs |
| 6 | **Export** | Download CSV logs, model checkpoints |

### Dashboard Pages

- **Home** — System overview, quick actions, recent activity
- **Training Manager** — Full training lifecycle (configure → train → monitor → compare → simulate)
- **Training Overview** — Episode reward curves, gating weights, degradation tracking
- **Agent Analysis** — Dominance stats, correlation matrices, routing visualization
- **Environment Sim** — Interactive wind turbine simulation
- **Reward Analysis** — Component breakdown, phase evolution, radar chart
- **Architecture** — Interactive Mermaid diagrams of all subsystems
- **Raw Data** — Full training log with export
- **Settings** — Backend connection, preferences, API reference

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | System status, device, active jobs |
| `GET` | `/api/workflow` | 6-step recommended workflow |
| `GET` | `/api/configs/` | List training configurations |
| `GET` | `/api/configs/{name}` | Get full config details |
| `PUT` | `/api/configs/{name}` | Update config fields |
| `POST` | `/api/configs/{name}/clone` | Clone a config |
| `POST` | `/api/training/start` | Start training job |
| `GET` | `/api/training/jobs` | List all jobs |
| `GET` | `/api/training/jobs/{id}` | Get job status |
| `POST` | `/api/training/jobs/{id}/cancel` | Cancel running job |
| `GET` | `/api/training/jobs/{id}/logs` | Get job logs |
| `GET` | `/api/runs/` | List completed runs |
| `GET` | `/api/runs/{name}` | Get run metrics (paginated) |
| `GET` | `/api/runs/{name}/csv` | Download training CSV |
| `GET` | `/api/runs/{name}/checkpoint` | Download model checkpoint |
| `GET` | `/api/runs/{a}/compare/{b}` | Compare two runs |
| `POST` | `/api/simulation/run` | Run environment simulation |

---

## iPAE: Interpretable Physics-Informed Autoencoder

The iPAE module provides interpretable degradation analysis through physics-constrained latent dimensions:

| Dim | Physical Meaning | Constraint | Activation |
|:---:|------------------|------------|------------|
| $z_1$ | Wear Index | Monotone ↑ | Softplus |
| $z_2$ | Thermal Drift | Bounded | Tanh |
| $z_3$ | Fatigue Accumulation | Monotone ↑ | Softplus |
| $z_4$ | Imbalance Severity | Oscillatory | Sin |
| $z_5$ | Lubrication State | Step-wise ↓ | Sigmoid |
| $z_6$ | Crack Propagation | Monotone ↑ | Softplus |
| $z_7$ | Misalignment | Bounded | Tanh |
| $z_8$ | Residual | None | Linear |

**Loss function:**

$$\mathcal{L}_{\text{iPAE}} = \mathcal{L}_{\text{recon}} + \lambda_p \mathcal{L}_{\text{ODE}} + \lambda_r \mathcal{L}_{\text{RUL}} + \lambda_m \mathcal{L}_{\text{mono}} + \lambda_{KL} \text{KL}(q \| p)$$

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.0+ |
| RL Environment | Gymnasium |
| GP Module | GPyTorch |
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Monitoring | TensorBoard |
| Scientific Computing | NumPy, SciPy, Pandas |

---

## Wind Turbine Environment

**State Space** ($\mathbb{R}^8$): Rotor speed, wind speed, generator torque, bearing temperature, power output, degradation index, grid demand, weather variability.

**Action Space** ($\mathbb{R}^3$): Pitch angle change (±15°), torque setpoint delta (±0.1), maintenance signal ([0,1]).

**Physics:** Aerodynamic torque via $C_p(\lambda, \beta)$, convective thermal dynamics, stress-dependent degradation accumulation. Critical threshold at $\delta = 0.9$.

---

## References

- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning," ICML 2018.
- Jacobs et al., "Adaptive Mixtures of Local Experts," Neural Computation, 1991.
- Raissi et al., "Physics-Informed Neural Networks," Journal of Computational Physics, 2019.
- Rasmussen & Williams, "Gaussian Processes for Machine Learning," MIT Press, 2006.

---

<div align="center">

**Predictive Analytics Core** · Siemens

</div>
