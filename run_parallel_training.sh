#!/bin/bash
# Parallel Training Execution Plan for HMARL-DRT
# =================================================
# Strategy:
#   - Phase 1a (PINN) and Phase 1c (PEARL) run in parallel (no dependencies)
#   - Phase 1b (GP) waits for PINN, Phase 1d (SAC) waits for PEARL
#   - Main training resumes from ep 500 with new unsupervised PEARL
#   - Validation runs in parallel with main training
#
# GPU: Quadro RTX 6000 (24GB) - can handle 2-3 processes simultaneously

set -e  # Exit on error
cd /home/ubuntu/hmarl-drt

LOG_DIR="runs/parallel_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "HMARL-DRT Parallel Training Execution"
echo "Log directory: $LOG_DIR"
echo "Started: $(date)"
echo "=========================================="

# Function to check GPU memory and wait if needed
wait_for_gpu() {
    local required_free=$1  # MB
    while true; do
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if [ "$free_mem" -gt "$required_free" ]; then
            break
        fi
        echo "Waiting for GPU memory (free: ${free_mem}MB, required: ${required_free}MB)..."
        sleep 10
    done
}

# ============================================================
# PHASE 0: Check existing checkpoints and decide strategy
# ============================================================

echo ""
echo "[Phase 0] Checking existing checkpoints..."

PINN_CKPT="runs/phase1a_pinn/pinn_final.pt"
GP_CKPT="runs/phase1b_gp/gp_final.pt"
PEARL_CKPT="runs/phase1c_pearl/pearl_final.pt"
SAC_CKPT="runs/phase1d_sac/sac_final.pt"
MAIN_CKPT="runs/checkpoints/ep_00500.pt"

SKIP_PINN=false
SKIP_GP=false
SKIP_PEARL=false
SKIP_SAC=false

if [ -f "$PINN_CKPT" ]; then
    echo "  ✓ PINN checkpoint found: $PINN_CKPT"
    SKIP_PINN=true
fi

if [ -f "$GP_CKPT" ]; then
    echo "  ✓ GP checkpoint found: $GP_CKPT"
    SKIP_GP=true
fi

if [ -f "$PEARL_CKPT" ]; then
    echo "  ✓ PEARL checkpoint found: $PEARL_CKPT"
    SKIP_PEARL=true
fi

if [ -f "$SAC_CKPT" ]; then
    echo "  ✓ SAC checkpoint found: $SAC_CKPT"
    SKIP_SAC=true
fi

if [ -f "$MAIN_CKPT" ]; then
    echo "  ✓ Main training checkpoint found: $MAIN_CKPT (ep 500)"
fi

# ============================================================
# PHASE 1: Parallel Pre-training (PINN + PEARL)
# ============================================================

echo ""
echo "[Phase 1] Starting parallel pre-training..."
echo "  - PINN and PEARL run in parallel (no dependencies)"
echo "  - GP waits for PINN, SAC waits for PEARL"

# Track PIDs
declare -a PIDS=()

# Phase 1a: PINN Pre-training (200 episodes)
if [ "$SKIP_PINN" = false ]; then
    echo ""
    echo "  [1a] Starting PINN pre-training (200 episodes)..."
    wait_for_gpu 8000  # Need ~8GB for PINN
    python3 -m hmarl_drt.pretrain_pinn --config hmarl_drt/configs/phase1a_pinn_pretrain.yaml \
        2>&1 | tee "$LOG_DIR/phase1a_pinn.log" &
    PINN_PID=$!
    PIDS+=($PINN_PID)
    echo "    PID: $PINN_PID"
else
    echo "  [1a] Skipping PINN (checkpoint exists)"
fi

# Phase 1c: PEARL Pre-training (500 episodes) - SUPERVISED VERSION
# NOTE: This is the OLD supervised PEARL. We'll use the NEW unsupervised PEARL
# during main training. This pretrain is optional.
if [ "$SKIP_PEARL" = false ]; then
    echo ""
    echo "  [1c] Starting PEARL pre-training (500 episodes, supervised)..."
    echo "    NOTE: This is supervised pretrain. Main training uses unsupervised PEARL."
    wait_for_gpu 8000
    python3 -m hmarl_drt.pretrain_pearl --config hmarl_drt/configs/phase1c_pearl_pretrain.yaml \
        2>&1 | tee "$LOG_DIR/phase1c_pearl.log" &
    PEARL_PID=$!
    PIDS+=($PEARL_PID)
    echo "    PID: $PEARL_PID"
else
    echo "  [1c] Skipping PEARL (checkpoint exists)"
fi

# Wait for PINN to finish before starting GP
if [ "$SKIP_PINN" = false ]; then
    echo ""
    echo "  Waiting for PINN to complete before starting GP..."
    wait $PINN_PID 2>/dev/null || true
    echo "  [1a] PINN completed: $(date)"
fi

# Phase 1b: GP Pre-training (200 episodes) - depends on PINN
if [ "$SKIP_GP" = false ]; then
    echo ""
    echo "  [1b] Starting GP pre-training (200 episodes)..."
    wait_for_gpu 8000
    python3 -m hmarl_drt.pretrain_gp --config hmarl_drt/configs/phase1b_gp_pretrain.yaml \
        2>&1 | tee "$LOG_DIR/phase1b_gp.log" &
    GP_PID=$!
    PIDS+=($GP_PID)
    echo "    PID: $GP_PID"
else
    echo "  [1b] Skipping GP (checkpoint exists)"
fi

# Wait for PEARL to finish before starting SAC
if [ "$SKIP_PEARL" = false ]; then
    echo ""
    echo "  Waiting for PEARL to complete before starting SAC..."
    wait $PEARL_PID 2>/dev/null || true
    echo "  [1c] PEARL completed: $(date)"
fi

# Phase 1d: SAC Pre-training (400 episodes) - depends on PEARL
if [ "$SKIP_SAC" = false ]; then
    echo ""
    echo "  [1d] Starting SAC pre-training (400 episodes)..."
    wait_for_gpu 10000
    python3 -m hmarl_drt.pretrain_sac --config hmarl_drt/configs/phase1d_sac_pretrain.yaml \
        2>&1 | tee "$LOG_DIR/phase1d_sac.log" &
    SAC_PID=$!
    PIDS+=($SAC_PID)
    echo "    PID: $SAC_PID"
else
    echo "  [1d] Skipping SAC (checkpoint exists)"
fi

# Wait for all Phase 1 to complete
echo ""
echo "  Waiting for all Phase 1 processes to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done

echo ""
echo "[Phase 1] All pre-training completed: $(date)"

# ============================================================
# PHASE 2-4: Main Training (Resume from ep 500)
# ============================================================

echo ""
echo "[Phase 2-4] Starting main training..."
echo "  Configuration: hmarl_drt/configs/wind_config.yaml"
echo "  Resume from: ep 500"
echo "  Target: 5000 episodes"
echo "  PEARL: UNSUPERVISED (dynamics prediction only)"

# Create a modified config to resume from ep 500
cat > hmarl_drt/configs/wind_config_resume.yaml <<EOF
# Resume training from episode 500 with unsupervised PEARL
# Auto-generated by run_parallel_training.sh

$(cat hmarl_drt/configs/wind_config.yaml)

training:
  warm_start_checkpoint: "runs/checkpoints/ep_00500.pt"
EOF

# Start main training
echo ""
echo "  Starting main training loop..."
wait_for_gpu 15000  # Need ~15GB for full HMARL system

python3 -m hmarl_drt.train --config hmarl_drt/configs/wind_config.yaml \
    2>&1 | tee "$LOG_DIR/main_training.log" &
MAIN_PID=$!

echo "    PID: $MAIN_PID"

# ============================================================
# VALIDATION: Run in parallel with training
# ============================================================

echo ""
echo "[Validation] Starting periodic validation..."

# Create validation script
cat > "$LOG_DIR/run_validation.sh" <<'VALIDATE_EOF'
#!/bin/bash
# Periodic validation during training
cd /home/ubuntu/hmarl-drt

LOG_DIR="$1"
EVAL_FREQ=50  # Episodes between evaluations

while true; do
    # Check if training is still running
    if ! pgrep -f "hmarl_drt.train" > /dev/null; then
        echo "Training completed, stopping validation loop"
        break
    fi
    
    # Get latest checkpoint
    LATEST_CKPT=$(ls -t runs/checkpoints/ep_*.pt 2>/dev/null | head -1)
    
    if [ -n "$LATEST_CKPT" ]; then
        EP_NUM=$(basename "$LATEST_CKPT" | sed 's/ep_\([0-9]*\).pt/\1/')
        
        # Run validation every EVAL_FREQ episodes
        if [ $((EP_NUM % EVAL_FREQ)) -eq 0 ]; then
            echo "[$(date)] Running validation at episode $EP_NUM"
            
            # Quick validation: run 4 episodes (one per regime)
            python3 -c "
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from hmarl_drt.envs.wind_turbine_env import WindTurbineEnv
from hmarl_drt.agents.sac_agent import SACAgent
from hmarl_drt.agents.gp_agent import GPAgent
from hmarl_drt.agents.pinn_agent import PINNAgent
from hmarl_drt.core.context_encoder import PEARLContextEncoder
import yaml

# Load config
with open('hmarl_drt/configs/wind_config.yaml') as f:
    cfg = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load checkpoint
ckpt = torch.load('$LATEST_CKPT', map_location=device)

# Quick test: can we load the model?
print(f'  Checkpoint ep {ckpt.get(\"episode\", \"unknown\")} loaded successfully')
print(f'  Keys: {list(ckpt.keys())[:5]}...')

# Test PEARL context encoder
if 'pearl_context_encoder' in ckpt:
    enc = PEARLContextEncoder(state_dim=8, action_dim=3, device=device)
    enc.load_state_dict(ckpt['pearl_context_encoder'])
    print(f'  PEARL encoder loaded: z_dim={enc.z_dim}')
elif 'pearl_encoder' in ckpt:
    print(f'  Old PEARL format detected (pearl_encoder)')
else:
    print(f'  WARNING: No PEARL encoder in checkpoint')
" 2>&1 | tee -a "$LOG_DIR/validation.log"
        fi
    fi
    
    sleep 60  # Check every minute
done
VALIDATE_EOF

chmod +x "$LOG_DIR/run_validation.sh"

# Start validation in background
"$LOG_DIR/run_validation.sh" "$LOG_DIR" &
VAL_PID=$!

echo "  Validation PID: $VAL_PID"

# ============================================================
# MONITORING: Track progress
# ============================================================

echo ""
echo "[Monitoring] Training progress will be logged to:"
echo "  - Main: $LOG_DIR/main_training.log"
echo "  - TensorBoard: runs/"
echo ""
echo "To monitor:"
echo "  tail -f $LOG_DIR/main_training.log"
echo "  tensorboard --logdir runs/ --port 6006"
echo ""
echo "=========================================="
echo "Training started. Press Ctrl+C to stop."
echo "=========================================="

# Wait for main training
wait $MAIN_PID

echo ""
echo "=========================================="
echo "Training completed: $(date)"
echo "=========================================="

# Stop validation
kill $VAL_PID 2>/dev/null || true

# Summary
echo ""
echo "Final checkpoints saved to: runs/checkpoints/"
ls -lh runs/checkpoints/*.pt | tail -5

echo ""
echo "Training logs saved to: $LOG_DIR"
