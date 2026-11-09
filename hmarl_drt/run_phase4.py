"""Run Phase 4 – Meta‑learning consolidation.

Phase 4 is the *consolidation* stage after the CEMRL integration (Phase 3).
Its purpose is to:
  1. Freeze parameters that have accumulated a large Fisher information
     (i.e., parameters that are important for previously‑seen regimes).
  2. Increase the Elastic‑Weight‑Consolidation (EWC) penalty so that future
     updates respect the frozen knowledge.
  3. Fine‑tune the gating network and the reward‑shaper for the remaining
     trainable parameters.

The existing training loop already supports EWC and respects a per‑parameter
Fisher matrix.  To implement the consolidation we simply raise the EWC
regularisation coefficient and tell the training script to *skip* any further
Fisher recomputation (the Fisher matrices from Phase 3 are kept in the
``EWCManager``).  The script below modifies the configuration accordingly
and runs a short demonstration (30 episodes).  For a full‑scale run the user
can increase ``training.n_episodes``.
"""

import yaml
import copy
import sys, os

# Ensure the project root is on ``sys.path`` so that imports resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hmarl_drt.train import train


def main() -> None:
    # Load the base configuration.
    with open("configs/wind_config.yaml") as f:
        cfg = yaml.safe_load(f)

    # ---------------------------------------------------------------------
    # Adjust training parameters for the consolidation phase.
    # ---------------------------------------------------------------------
    cfg["training"] = copy.deepcopy(cfg.get("training", {}))
    cfg["training"]["n_episodes"] = 30  # short run for demonstration
    cfg["training"]["eval_freq"] = 5
    cfg["training"]["save_freq"] = 1000

    # ---------------------------------------------------------------------
    # Consolidation‑specific tweaks.
    # ---------------------------------------------------------------------
    # 1. Increase the EWC penalty (lambda) – this makes the optimiser respect
    #    the previously‑computed Fisher information more strongly.
    cfg.setdefault("cemrl", {})
    cfg["cemrl"].setdefault("ewc", {})
    cfg["cemrl"]["ewc"]["lambda_base"] = 800.0  # double the Phase 3 value

    # 2. Disable further Fisher recomputation – we want to keep the Fisher
    #    matrices that were collected during Phase 3.
    cfg["cemrl"]["ewc"]["fisher_samples"] = 0

    # 3. Optionally, we could ask the ``EWCManager`` to freeze parameters
    #    whose Fisher magnitude exceeds a threshold.  The manager already
    #    implements ``freeze_high_fisher`` (see ``core/ewc_manager.py``).  We
    #    enable it via a flag in the config.
    cfg["cemrl"]["ewc"]["freeze_high_fisher"] = True
    cfg["cemrl"]["ewc"]["fisher_threshold"] = 0.05  # example threshold

    # 4. Keep CEMRL enabled – we still want the meta‑gating network to be
    #    fine‑tuned, but with the stronger EWC regulariser.
    cfg["cemrl"]["enabled"] = True

    # ---------------------------------------------------------------------
    # Write a temporary config file and launch training.
    # ---------------------------------------------------------------------
    tmp_path = "tmp_phase4.yaml"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(cfg, f)

    print("Running Phase 4 (meta‑learning consolidation) with", cfg["training"]["n_episodes"], "episodes")
    train(tmp_path)


if __name__ == "__main__":
    main()
