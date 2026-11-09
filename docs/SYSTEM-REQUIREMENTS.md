# CEMRL-HMARL-DRT — System Requirements & Resource Usage

**Date:** 2026-09-24
**Project:** Context-aware Ensemble Model-based Reinforcement Learning for
Degradation-Robust Turbine control (CEMRL-HMARL-DRT)

---

## 1. Hardware Requirements

### 1.1 Minimum Hardware (used for this project)

| Component | Specification | Notes |
|-----------|---------------|-------|
| **GPU** | NVIDIA Quadro RTX 6000, 24 GB VRAM | CUDA-capable, ≥ 8 GB VRAM recommended |
| **CPU** | 4 cores | Single-environment RL loop is CPU-light |
| **RAM** | 8 GB (7 GB usable) | ~4 GB free during training |
| **Storage** | ~400 MB per full run | 52 checkpoints + logs + CSV |
| **OS** | Linux (Ubuntu) | CUDA 11.x+ |

### 1.2 Recommended Hardware

| Component | Specification | Rationale |
|-----------|---------------|-----------|
| **GPU** | ≥ 16 GB VRAM, modern (RTX 30xx/40xx, A100) | Faster training, higher utilisation |
| **CPU** | ≥ 8 cores | Enables vectorised environments (future work) |
| **RAM** | ≥ 16 GB | Larger replay buffers, parallel baselines |
| **Storage** | ≥ 10 GB free | Multiple runs + validation artifacts |

### 1.3 Software Requirements

| Component | Version |
|-----------|---------|
| Python | 3.10+ |
| PyTorch | 2.x (CUDA build) |
| NumPy | 1.24+ |
| PyYAML | 6.x |
| Matplotlib | 3.x (for report figures) |
| CUDA Toolkit | 11.x+ (if GPU) |
| TensorBoard | optional (logging) |

---

## 2. Compute Time Breakdown

### 2.1 Training Runs

| Task | Episodes | Wall-clock | GPU-hours | CPU-hours |
|------|----------|------------|-----------|-----------|
| **Main CEMRL training** (5000 ep) | 5000 | **95.5 hours** | ~95.5 | ~95.5 |
| no-CEMRL baseline R0 (300 ep) | 300 | 2.7 hours | 2.7 | 2.7 |
| no-CEMRL baseline R1 (300 ep) | 300 | 2.7 hours | 2.7 | 2.7 |
| no-CEMRL baseline R2 (300 ep) | 300 | 2.7 hours | 2.7 | 2.7 |
| no-CEMRL baseline R3 (300 ep) | 300 | 1.7 hours | 1.7 | 1.7 |
| **All 4 baselines (parallel)** | 1200 | **2.7 hours** | 10.8 | 10.8 |

**Training subtotal: ~98 hours wall-clock** (main run + parallel baselines).

### 2.2 Validation Runs

| Task | Scope | Wall-clock |
|------|-------|------------|
| Checkpoint validation (50 ckpts) | 50 × 2 ranges × 10 ep | ~1.25 hours |
| Degradation deep-dive (ep_2300) | 16 modes × 2 ranges × 10 ep | ~0.5 hours |
| Adaptation speed v2 | 4 baselines + 4 CEMRL × 30 ep | ~1.0 hour |
| Regime-transition analysis | 16 modes × 2 ranges × 10 ep | ~0.5 hours |

**Validation subtotal: ~3.25 hours.**

### 2.3 Total

| Category | Wall-clock | GPU-hours |
|----------|------------|-----------|
| Training | ~98 hours | ~106 |
| Validation | ~3.25 hours | ~3.25 |
| **Grand total** | **~101 hours (~4.2 days)** | **~109 GPU-hours** |

---

## 3. Per-Episode Timing

| Task | Time per episode |
|------|------------------|
| Main CEMRL training | 68.7 s/episode |
| no-CEMRL baseline | ~32 s/episode |
| Validation episode | ~30 s/episode |

**Why ~69 s/episode:** Each episode is 1000 steps. Each step runs:
- 4 agent forward passes (SAC, iPAE BiLSTM, FM, PINN)
- Shared embedding fusion
- CEMRL arbiter gating
- iPAE uncertainty (MC-dropout)
- Environment physics step

The iPAE BiLSTM encoder is the main cost (sequential over the window).

---

## 4. GPU Utilisation

| Configuration | GPU utilisation | Notes |
|---------------|-----------------|-------|
| batch_size 128, window 20, lstm 64 | 7-8% | launch-latency-bound |
| batch_size 512, window 10, lstm 128 | 11-14% | larger tensors, more occupancy |
| batch_size 256, window 10, lstm 64 | 5-7% | current config |

**Key insight:** The workload is **launch-latency-bound, not compute-bound**.
A single environment does batch-1 inference every timestep, so the GPU idles
between thousands of tiny kernel launches. GPU utilisation is inherently low
(~5-14%) for this architecture.

**To increase utilisation:**
1. **Vectorised environments** (8 envs → batch-8 inference) — the real fix,
   ~8× throughput, but requires rewriting the training loop
2. Larger batch_size (marginal, tested)
3. Larger iPAE hidden size (marginal, tested)

---

## 5. Memory Requirements

| Component | Memory |
|-----------|--------|
| Model weights (all agents) | ~50 MB |
| Replay buffer (1M transitions × 8-dim) | ~32 MB |
| Checkpoint (per 100 ep) | ~1.9 MB |
| Full run (52 checkpoints + logs) | ~378 MB |
| GPU VRAM | ~534 MB (of 24 GB) |

**The system is very memory-light** — the bottleneck is compute time, not memory.

---

## 6. Storage Requirements

| Artifact | Size |
|----------|------|
| Main run (52 checkpoints + CSV + log) | 378 MB |
| Per-regime baseline (300 ep) | ~10 MB each |
| Validation JSON results | ~1 MB |
| Report figures | ~1 MB |
| **Total per full experiment** | **~450 MB** |

---

## 7. Scaling Estimates

### 7.1 If you change episode length

| max_steps | Time per episode | 5000-episode run |
|-----------|------------------|------------------|
| 1000 (current) | 68.7 s | 95.5 hours |
| 500 | ~34 s | ~48 hours |
| 300 | ~21 s | ~29 hours |

### 7.2 If you change number of episodes

| Episodes | Wall-clock (at 68.7 s/ep) |
|----------|---------------------------|
| 1000 | 19 hours |
| 2000 | 38 hours |
| 3000 | 57 hours |
| 5000 (current) | 95.5 hours |

### 7.3 If you add vectorised environments (8×)

| Configuration | Estimated 5000-episode run |
|---------------|----------------------------|
| Current (single env) | 95.5 hours |
| 8 vectorised envs | ~12-15 hours |

---

## 8. Requirements Checklist

### To reproduce the full experiment, you need:

- [ ] **Hardware:** GPU with ≥ 8 GB VRAM, 4+ CPU cores, 8+ GB RAM, ~1 GB storage
- [ ] **Software:** Python 3.10+, PyTorch 2.x (CUDA), NumPy, PyYAML, Matplotlib
- [ ] **Time:** ~101 hours wall-clock (~4.2 days) for training + validation
- [ ] **Compute:** ~109 GPU-hours
- [ ] **Config:** `hmarl_drt/configs/wind_fresh_ipae_supervised.yaml`
- [ ] **FM API key:** for the foundation-model agent (optional, falls back to heuristics)

### To run just the main training (no validation):

- [ ] **Time:** ~95.5 hours
- [ ] **Compute:** ~95.5 GPU-hours
- [ ] **Storage:** ~378 MB

### To run just validation on existing checkpoints:

- [ ] **Time:** ~3.25 hours
- [ ] **Compute:** ~3.25 GPU-hours
- [ ] **Storage:** ~1 MB (results)

---

## 9. Commands

### Training
```bash
# Main CEMRL training (5000 episodes)
cd hmarl_drt
python train.py configs/wind_fresh_ipae_supervised.yaml

# no-CEMRL baselines (per regime, for adaptation comparison)
python train.py configs/wind_baseline_nocemrl_R0.yaml
python train.py configs/wind_baseline_nocemrl_R1.yaml
python train.py configs/wind_baseline_nocemrl_R2.yaml
python train.py configs/wind_baseline_nocemrl_R3.yaml
```

### Validation
```bash
# Checkpoint validation (all 50 checkpoints)
python validate_checkpoints.py

# Degradation deep-dive on best checkpoint
python validate_ep2300_deepdive.py --n-episodes 10

# Adaptation speed (CEMRL vs no-CEMRL baselines)
python evaluate_regime_transition_adaptation_v2.py --baseline-episode 300
```

### Report generation
```bash
# Generate all report figures
python generate_report_figures.py
```

---

## 10. Notes & Caveats

1. **GPU utilisation is low (5-14%)** by design — the workload is
   launch-latency-bound. This is not a bug; it's inherent to single-environment
   RL with a sequence model (iPAE BiLSTM).

2. **The 95.5-hour training run** was done on a Quadro RTX 6000 (24 GB). A
   newer GPU (RTX 4090, A100) would be faster per-step but the bottleneck is
   launch latency, so the speedup is modest (~1.5-2×).

3. **The FM agent** makes API calls to a Siemens LLM gateway. If the API is
   unreachable, it falls back to rule-based heuristics (no training impact, but
   the reward-weight suggestions become static).

4. **Memory is not a constraint** — the system uses ~534 MB VRAM and ~4 GB RAM.
   The limiting factor is wall-clock time.

5. **For faster iteration**, reduce `max_steps` (e.g., 300 → 29 hours for 5000
   episodes) or reduce `n_episodes`. The system converges by ~2000 episodes, so
   a 2000-episode run (~38 hours) captures most of the learning.
