#!/usr/bin/env python3
"""Analyze the three connectome datasets and demonstrate the fusion pipeline.

This script shows the full pipeline from raw data to a trainable network:
1. Load each dataset
2. Identify bridge neurons
3. Fuse into a single graph
4. Build the trainable network
5. Run a smoke test forward pass

Can run in two modes:
- "demo": Uses synthetic data to demonstrate the architecture (no downloads)
- "full": Downloads and processes real connectome data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fly_connectome.data.bridge import classify_network_role
from fly_connectome.network.graph import FusedConnectomeGraph, build_fused_graph
from fly_connectome.network.model import ConnectomeNetwork


# ═══════════════════════════════════════════════════════════════════
# Dataset Analysis Report
# ═══════════════════════════════════════════════════════════════════

DATASET_REPORT = """
╔══════════════════════════════════════════════════════════════════════╗
║               FLY CONNECTOME FUSION — DATASET ANALYSIS              ║
╚══════════════════════════════════════════════════════════════════════╝

Three connectome datasets to fuse into a single trainable neural network:

┌─────────────────────────────────────────────────────────────────────┐
│ 1. FAFB — Female Adult Fly Brain (FlyWire)                         │
│    ├─ Neurons:     ~139,255 proofread                              │
│    ├─ Synapses:    ~50 million connections                         │
│    ├─ Coverage:    Entire brain (both hemispheres)                 │
│    ├─ Includes:    Central brain + optic lobes                     │
│    ├─ NT data:     ACh, GABA, Glu, Octopamine predictions         │
│    ├─ Source:      Dorkenwald et al., Nature 2024                  │
│    └─ Access:      Zenodo, Codex, CAVE API, fafbseg-py            │
├─────────────────────────────────────────────────────────────────────┤
│ 2. MANC — Male Adult Nerve Cord (Janelia FlyEM)                    │
│    ├─ Neurons:     ~23,000 traced                                  │
│    ├─ Presynaptic: ~10 million sites                               │
│    ├─ Postsynaptic: ~74 million densities                          │
│    ├─ Coverage:    Complete VNC (T1-T3 + abdominal)                │
│    ├─ NT data:     ACh, GABA, Glu predictions per neuron           │
│    ├─ Source:      Takemura et al., eLife 2024                     │
│    └─ Access:      neuPrint, Google Cloud flat files               │
├─────────────────────────────────────────────────────────────────────┤
│ 3. Male Optic Lobe — Right hemisphere (Janelia FlyEM)              │
│    ├─ Neurons:     ~50,000+                                        │
│    ├─ Cell types:  700+ distinct types                             │
│    ├─ Coverage:    Complete right optic lobe                       │
│    │               (Lamina, Medulla, Lobula, Lobula Plate)         │
│    ├─ Source:      Nern et al., Nature 2025                        │
│    └─ Access:      neuPrint (optic-lobe:v1.0), Google Cloud        │
└─────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════╗
║                    FUSION STRATEGY                                   ║
╚══════════════════════════════════════════════════════════════════════╝

The datasets connect through three types of bridge neurons:

    OPTIC LOBE ──VPNs──▶ CENTRAL BRAIN ──DNs──▶ VNC (MANC)
         ▲                     ▲                    │
         │                     └────ANs─────────────┘
    Photoreceptors                              Motor Neurons
    (R1-R8)                                    (wings, legs, etc.)

Bridge neuron types:
  • Visual Projection Neurons (VPNs): OL → Central Brain
    LC, LPLC, LT, ML, VS, HS types (~100+ types)
  • Descending Neurons (DNs): Brain → VNC
    ~350 cell types, ~1,100 individual neurons
  • Ascending Neurons (ANs): VNC → Brain
    Proprioceptive & mechanosensory feedback

Pre-existing resource: The BANC project "Frankenbrain" SQLite database
already fuses FAFB + MANC via matched neck connective neurons.
We augment this with the detailed optic lobe circuitry.

╔══════════════════════════════════════════════════════════════════════╗
║               TRAINABLE NETWORK ARCHITECTURE                         ║
╚══════════════════════════════════════════════════════════════════════╝

Following Lappalainen et al. (Nature 2024, "flyvis"):

1. TOPOLOGY: Fixed from connectome (sparse, ~200K+ neurons)
2. LEARNABLE PARAMS:
   - Synapse strengths: 1 param per edge (~millions)
   - Neuron time constants: 1 param per neuron (or per cell type)
   - Neuron biases: 1 param per neuron
3. CONSTRAINTS:
   - Dale's law: sign(output) fixed by neurotransmitter type
   - Sparsity: only connectome-present edges allowed
   - Weight positivity: softplus parameterization
4. DYNAMICS: Either rate-based (simpler) or LIF (biologically richer)
5. TRAINING: Backpropagation through time (BPTT) for supervised tasks,
   or policy gradient / PPO for RL-based sensorimotor control

Total expected parameters for full fused network:
  - Synapse weights: ~10-50M (one per biological synapse bundle)
  - Neuron dynamics:  ~400K  (tau + bias per neuron)
  - Much smaller than unconstrained network of same size
    (which would need n^2 = ~40 billion parameters)
"""


def create_demo_datasets():
    """Create synthetic datasets that mirror the structure of real data.

    This generates small-scale versions of FAFB, MANC, and optic lobe
    with realistic neuron type distributions and connectivity patterns.
    """
    np.random.seed(42)

    # ── Optic Lobe (scaled down) ──
    # Real: ~50K neurons. Demo: 500
    ol_types = (
        ["R1"] * 20 + ["R6"] * 20 + ["R7"] * 10 + ["R8"] * 10 +  # photoreceptors
        ["L1"] * 20 + ["L2"] * 20 + ["L3"] * 10 + ["L5"] * 10 +  # lamina
        ["Mi1"] * 15 + ["Mi4"] * 15 + ["Mi9"] * 15 +  # medulla intrinsic
        ["Tm1"] * 15 + ["Tm2"] * 15 + ["Tm3"] * 15 + ["Tm9"] * 10 +  # transmedullary
        ["T4a"] * 15 + ["T4b"] * 15 + ["T5a"] * 15 + ["T5b"] * 15 +  # motion detectors
        ["LC10"] * 10 + ["LC11"] * 10 + ["LPLC1"] * 10 + ["LPLC2"] * 10 +  # VPNs
        ["LT1"] * 5 + ["VS"] * 10 + ["HS"] * 10 +  # tangential cells
        ["OL_inter"] * 130  # other interneurons
    )
    n_ol = len(ol_types)
    ol_nts = ["acetylcholine"] * n_ol  # Most OL neurons are cholinergic
    for i, t in enumerate(ol_types):
        if t.startswith("L") or t.startswith("Mi9"):
            ol_nts[i] = "glutamate"  # Some lamina/medulla cells are glutamatergic
        if t in ("T4a", "T4b", "T5a", "T5b"):
            ol_nts[i] = "acetylcholine"  # T4/T5 are cholinergic

    ol_neurons = pd.DataFrame({
        "id": [f"ol_{i}" for i in range(n_ol)],
        "cell_type": ol_types,
        "nt_type": ol_nts,
        "source": "optic_lobe",
    })

    # ── FAFB Central Brain (scaled down) ──
    # Real: ~90K non-OL neurons. Demo: 800
    cb_types = (
        ["KC"] * 80 +         # Kenyon cells (mushroom body)
        ["MBON"] * 20 +       # MB output neurons
        ["MBIN"] * 15 +       # MB input neurons
        ["DAN"] * 15 +        # Dopaminergic neurons
        ["EPG"] * 10 +        # Central complex (navigation)
        ["PEN"] * 10 +
        ["PFL"] * 10 +
        ["hDelta"] * 10 +
        ["DN"] * 40 +         # Descending neurons (bridge to VNC)
        ["AN_brain"] * 20 +   # Ascending neuron targets
        ["AVLP_inter"] * 100 + # Visual association
        ["SLP_inter"] * 80 +
        ["LH_inter"] * 60 +   # Lateral horn (innate behavior)
        ["CB_inter"] * 330    # Other central brain interneurons
    )
    n_cb = len(cb_types)
    cb_nts = []
    for t in cb_types:
        if t in ("KC", "EPG", "PEN", "PFL", "MBON"):
            cb_nts.append("acetylcholine")
        elif t.startswith("DAN"):
            cb_nts.append("dopamine")
        elif t == "hDelta":
            cb_nts.append("gaba")
        elif t.startswith("DN"):
            cb_nts.append(np.random.choice(["acetylcholine", "gaba", "glutamate"]))
        else:
            cb_nts.append(np.random.choice(
                ["acetylcholine", "gaba", "glutamate"], p=[0.5, 0.3, 0.2]
            ))

    cb_neurons = pd.DataFrame({
        "id": [f"cb_{i}" for i in range(n_cb)],
        "cell_type": cb_types,
        "nt_type": cb_nts,
        "source": "fafb",
    })

    # ── MANC VNC (scaled down) ──
    # Real: ~23K neurons. Demo: 300
    vnc_types = (
        ["MN_wing"] * 20 +     # Wing motor neurons
        ["MN_leg"] * 30 +      # Leg motor neurons (6 legs)
        ["MN_haltere"] * 5 +   # Haltere motor neurons
        ["MN_neck"] * 5 +      # Neck motor neurons
        ["AN"] * 25 +          # Ascending neurons (bridge to brain)
        ["SN_prop"] * 20 +     # Proprioceptive sensory neurons
        ["SN_mech"] * 15 +     # Mechanosensory neurons
        ["CPG_wing"] * 20 +    # Wing CPG interneurons
        ["CPG_leg"] * 20 +     # Leg CPG interneurons
        ["premotor"] * 50 +    # Premotor interneurons
        ["VNC_inter"] * 90     # Other VNC interneurons
    )
    n_vnc = len(vnc_types)
    vnc_nts = []
    for t in vnc_types:
        if t.startswith("MN"):
            vnc_nts.append("acetylcholine")  # MNs at NMJ are cholinergic
        elif t.startswith("SN"):
            vnc_nts.append("acetylcholine")
        elif t.startswith("CPG"):
            vnc_nts.append(np.random.choice(["acetylcholine", "gaba", "glutamate"]))
        else:
            vnc_nts.append(np.random.choice(
                ["acetylcholine", "gaba", "glutamate"], p=[0.4, 0.35, 0.25]
            ))

    vnc_neurons = pd.DataFrame({
        "id": [f"vnc_{i}" for i in range(n_vnc)],
        "cell_type": vnc_types,
        "nt_type": vnc_nts,
        "source": "manc",
    })

    # ── Generate connectivity ──
    all_neurons = pd.concat([ol_neurons, cb_neurons, vnc_neurons], ignore_index=True)
    n_total = len(all_neurons)
    id_to_idx = {nid: i for i, nid in enumerate(all_neurons["id"])}

    edges_list = []

    def add_edges(src_mask, tgt_mask, density, min_syn=5, max_syn=50):
        """Generate random edges between neuron groups."""
        src_ids = all_neurons.loc[src_mask, "id"].values
        tgt_ids = all_neurons.loc[tgt_mask, "id"].values
        for s in src_ids:
            n_targets = max(1, int(len(tgt_ids) * density))
            targets = np.random.choice(tgt_ids, size=n_targets, replace=False)
            for t in targets:
                if s != t:
                    edges_list.append({
                        "pre_id": s, "post_id": t,
                        "syn_count": np.random.randint(min_syn, max_syn),
                    })

    # Optic lobe internal: R→L→M→Lo→LP (feedforward with recurrence)
    is_ol = all_neurons["source"] == "optic_lobe"
    is_R = all_neurons["cell_type"].str.match(r"^R\d")
    is_L = all_neurons["cell_type"].str.startswith("L")
    is_Mi = all_neurons["cell_type"].str.startswith("Mi")
    is_Tm = all_neurons["cell_type"].str.startswith("Tm")
    is_T45 = all_neurons["cell_type"].str.startswith("T4") | all_neurons["cell_type"].str.startswith("T5")
    is_VPN = all_neurons["cell_type"].str.match(r"^(LC|LPLC|LT|VS|HS)")

    add_edges(is_R, is_L, 0.3, 10, 30)       # photoreceptors → lamina
    add_edges(is_L, is_Mi, 0.2, 8, 25)        # lamina → medulla
    add_edges(is_Mi, is_Tm, 0.15, 5, 20)      # medulla intrinsic → transmedullary
    add_edges(is_Tm, is_T45, 0.2, 10, 30)     # transmedullary → T4/T5
    add_edges(is_T45, is_VPN, 0.15, 5, 15)    # T4/T5 → visual projection neurons

    # VPNs → Central Brain (the OL-CB bridge)
    is_cb = all_neurons["source"] == "fafb"
    is_AVLP = all_neurons["cell_type"].str.contains("AVLP")
    is_LH = all_neurons["cell_type"].str.contains("LH")
    add_edges(is_VPN, is_AVLP | is_LH, 0.1, 5, 15)

    # Central brain internal connectivity
    is_KC = all_neurons["cell_type"] == "KC"
    is_MBON = all_neurons["cell_type"] == "MBON"
    is_MBIN = all_neurons["cell_type"] == "MBIN"
    is_DAN = all_neurons["cell_type"] == "DAN"
    is_CX = all_neurons["cell_type"].str.match(r"^(EPG|PEN|PFL|hDelta)")
    is_DN = all_neurons["cell_type"] == "DN"

    add_edges(is_MBIN, is_KC, 0.05, 3, 10)    # MBIN → KC
    add_edges(is_KC, is_MBON, 0.03, 5, 15)    # KC → MBON
    add_edges(is_DAN, is_KC, 0.05, 3, 8)      # DAN → KC (learning signal)
    add_edges(is_MBON, is_DN, 0.1, 5, 15)     # MBON → DN (decision → action)
    add_edges(is_CX, is_DN, 0.1, 5, 15)       # CX → DN (navigation → action)
    add_edges(is_cb & ~is_DN, is_cb, 0.02, 3, 12)  # recurrent CB

    # DNs → VNC (the brain-VNC bridge)
    is_vnc = all_neurons["source"] == "manc"
    is_premotor = all_neurons["cell_type"].str.contains("premotor|CPG")
    add_edges(is_DN, is_premotor, 0.15, 8, 25)

    # VNC internal: premotor → motor, CPG dynamics
    is_MN = all_neurons["cell_type"].str.startswith("MN")
    is_CPG = all_neurons["cell_type"].str.startswith("CPG")
    is_AN = all_neurons["cell_type"] == "AN"
    is_SN = all_neurons["cell_type"].str.startswith("SN")

    add_edges(is_premotor | is_CPG, is_MN, 0.15, 10, 30)
    add_edges(is_CPG, is_CPG, 0.2, 5, 15)     # CPG recurrence
    add_edges(is_SN, is_vnc & ~is_MN, 0.05, 3, 10)  # sensory → interneurons

    # ANs → Central Brain (the VNC-brain bridge, feedback)
    add_edges(is_AN, is_cb, 0.05, 5, 15)

    edges = pd.DataFrame(edges_list)
    edges["weight"] = edges["syn_count"].astype(float)

    return all_neurons, edges


def run_demo():
    """Run the full pipeline with synthetic data."""
    print(DATASET_REPORT)

    print("\n" + "=" * 70)
    print("RUNNING DEMO WITH SYNTHETIC DATA")
    print("=" * 70)

    # Create synthetic datasets
    print("\n[1/5] Creating synthetic datasets...")
    neurons, edges = create_demo_datasets()
    print(f"  Total neurons: {len(neurons):,}")
    print(f"  Total edges:   {len(edges):,}")

    # Print neuron distribution
    print("\n  Neurons by source:")
    for src, count in neurons["source"].value_counts().items():
        print(f"    {src}: {count:,}")

    print("\n  Neurons by cell type category:")
    for ct, count in neurons["cell_type"].value_counts().head(15).items():
        print(f"    {ct}: {count:,}")

    # Classify roles
    print("\n[2/5] Classifying functional roles...")
    roles = classify_network_role(neurons)
    for role, count in roles.value_counts().items():
        print(f"    {role}: {count:,}")

    # Build fused graph
    print("\n[3/5] Building fused connectome graph...")
    graph = build_fused_graph(neurons, edges, roles, min_syn=3)
    print(graph.summary())

    # Build trainable network
    print("\n[4/5] Building trainable ConnectomeNetwork...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConnectomeNetwork(graph, neuron_type="rate", device=device)
    print(model.parameter_summary())

    # Smoke test
    print("\n[5/5] Running smoke test forward pass...")
    batch_size = 2
    n_steps = 50
    n_inputs = model.n_inputs
    n_outputs = model.n_outputs

    print(f"  Batch size: {batch_size}")
    print(f"  Time steps: {n_steps}")
    print(f"  Input dim:  {n_inputs} (photoreceptors + sensory)")
    print(f"  Output dim: {n_outputs} (motor neurons)")

    # Random input signal
    inputs = torch.randn(batch_size, n_steps, n_inputs, device=device)
    outputs = model(inputs, n_steps=n_steps)

    print(f"  Output shape: {outputs.shape}")
    print(f"  Output mean:  {outputs.mean().item():.4f}")
    print(f"  Output std:   {outputs.std().item():.4f}")

    # Test gradient flow
    print("\n  Testing gradient flow (backprop)...")
    loss = outputs.sum()
    loss.backward()

    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {n_grad}/{n_total}")

    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    print(f"  Mean gradient norm: {np.mean(grad_norms):.6f}")
    print(f"  Max gradient norm:  {np.max(grad_norms):.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("FUSION ARCHITECTURE SUMMARY")
    print("=" * 70)
    print("""
The fused connectome network has three processing stages:

  STAGE 1 — Visual Processing (Optic Lobe)
    Photoreceptors (R1-R8) → Lamina (L1-L5) → Medulla (Mi, Tm) →
    Lobula/LP (T4/T5) → Visual Projection Neurons (LC, LPLC, VS, HS)

  STAGE 2 — Central Processing (FAFB Brain)
    VPNs → AVLP/SLP (visual association) → Mushroom Body (learning) →
    Central Complex (navigation) → Descending Neurons

  STAGE 3 — Motor Control (MANC VNC)
    DNs → Premotor interneurons → CPGs → Motor Neurons
    (+ sensory feedback via ascending neurons back to brain)

Key architectural properties:
  • Sparse: only connectome-present edges (~0.01% density)
  • Constrained: Dale's law, topology fixed
  • Learnable: synapse weights, time constants, biases
  • Recurrent: multiple feedback loops at all levels
  • Modular: clear functional decomposition

Next steps for training to fly:
  1. Connect to a physics simulator (e.g. MuJoCo Drosophila model)
  2. Define reward: stay airborne, navigate to targets
  3. Train with PPO/SAC using motor neuron outputs as actions
  4. Visual input from simulated compound eye
  5. Proprioceptive input from simulated joint angles
""")


def run_full(data_dir: str):
    """Run with real connectome data (requires downloads)."""
    from fly_connectome.data.download import download_fafb
    from fly_connectome.data.loaders import load_fafb_from_feather

    data_path = Path(data_dir)
    print("Downloading FAFB dataset...")
    fafb_dir = download_fafb(data_path)
    print("Loading FAFB...")
    fafb_data = load_fafb_from_feather(fafb_dir)
    print(fafb_data.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fly connectome fusion analysis")
    parser.add_argument(
        "--mode", choices=["demo", "full"], default="demo",
        help="demo: synthetic data; full: download real data"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory for downloaded data"
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    else:
        run_full(args.data_dir)
