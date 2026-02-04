"""Bridge logic for connecting the three connectome datasets.

The three datasets form a complete sensorimotor loop:

    Photoreceptors (R1-R8)
        │
        ▼
    ┌─────────────────────┐
    │   OPTIC LOBE        │  ~50K neurons
    │   (visual processing)│  Lamina → Medulla → Lobula → Lobula Plate
    └────────┬────────────┘
             │  Visual Projection Neurons (VPNs)
             │  LC, LPLC, LT, ML types
             ▼
    ┌─────────────────────┐
    │   FAFB CENTRAL BRAIN│  ~90K neurons (non-OL portion)
    │   Mushroom Body      │  Decision-making, learning
    │   Central Complex    │  Navigation, motor planning
    │   Lateral Horn       │  Innate behaviour
    └────────┬────────────┘
             │  Descending Neurons (DNs)
             │  ~350 cell types, ~1100 neurons
             ▼
    ┌─────────────────────┐
    │   MANC (VNC)        │  ~23K neurons
    │   Motor neurons     │  Wing, leg, haltere, neck
    │   Premotor circuits │  CPGs, reflexes
    └────────┬────────────┘
             │  Motor output (wings, legs, etc.)
             │
             │  Ascending Neurons (ANs)
             │  Proprioception, mechanosensation
             ▼
        Back to Central Brain (feedback loop)

Bridging strategy
=================
1. FAFB ↔ MANC: Already connected in frankenbrain via BANC neck bridge.
   The neck connective contains:
   - Descending Neurons (DNs): brain → VNC commands
   - Ascending Neurons (ANs): VNC → brain feedback
   - Sensory Ascending (SAs): sensory organs → brain

2. Optic Lobe → FAFB Central Brain: Connected via Visual Projection Neurons.
   These are neurons intrinsic to FAFB but whose cell bodies/arbors span
   both the optic lobe neuropils and central brain neuropils.

3. The Male Optic Lobe dataset (Nern et al.) neurons are cross-matched to
   FAFB optic lobe neurons via the flyconnectome/ol_annotations repository.
   Cell types are conserved across sexes.

For our trainable network, we use two bridging approaches:
- Approach A (recommended): Use frankenbrain + augment with optic lobe detail
- Approach B: Load all three separately and bridge with matched cell types
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fly_connectome.data.loaders import ConnectomeData


@dataclass
class BridgeNeurons:
    """Neurons that connect two datasets."""

    source_dataset: str
    target_dataset: str
    bridge_type: str          # "descending", "ascending", "visual_projection"
    neuron_pairs: pd.DataFrame  # columns: source_id, target_id, cell_type, confidence


def identify_descending_neurons(fafb: ConnectomeData) -> pd.DataFrame:
    """Identify descending neurons (DNs) in the FAFB dataset.

    DNs are brain neurons that project to the VNC via the neck connective.
    They carry motor commands from the brain to motor circuits.

    Known DN classes:
    - DNa: anterior descending
    - DNb: bilateral descending
    - DNg: giant fiber (escape response)
    - DNp: posterior descending
    """
    dn_mask = pd.Series(False, index=fafb.neurons.index)

    if "cell_type" in fafb.neurons.columns:
        ct = fafb.neurons["cell_type"].fillna("")
        dn_mask |= ct.str.startswith("DN")

    if "super_class" in fafb.neurons.columns:
        sc = fafb.neurons["super_class"].fillna("")
        dn_mask |= sc.str.contains("descending", case=False)

    return fafb.neurons[dn_mask].copy()


def identify_ascending_neurons(manc: ConnectomeData) -> pd.DataFrame:
    """Identify ascending neurons (ANs) in the MANC dataset.

    ANs carry sensory/proprioceptive information from VNC back to brain.
    """
    an_mask = pd.Series(False, index=manc.neurons.index)

    if "cell_type" in manc.neurons.columns:
        ct = manc.neurons["cell_type"].fillna("")
        an_mask |= ct.str.startswith("AN")

    if "super_class" in manc.neurons.columns:
        sc = manc.neurons["super_class"].fillna("")
        an_mask |= sc.str.contains("ascending", case=False)

    return manc.neurons[an_mask].copy()


def identify_visual_projection_neurons(fafb: ConnectomeData) -> pd.DataFrame:
    """Identify visual projection neurons (VPNs) in FAFB.

    VPNs connect the optic lobe to the central brain.
    Key types:
    - LC (Lobula Columnar): feature detectors → central brain
    - LPLC (Lobula Plate-Lobula Columnar): motion → central brain
    - LT (Lobula Tangential)
    - ML (Medulla → Lobula)
    - VCH, DCH (ventral/dorsal centrifugal horizontal)
    """
    vpn_prefixes = ["LC", "LPLC", "LT", "ML", "VCH", "DCH", "VS", "HS"]

    vpn_mask = pd.Series(False, index=fafb.neurons.index)
    if "cell_type" in fafb.neurons.columns:
        ct = fafb.neurons["cell_type"].fillna("")
        for prefix in vpn_prefixes:
            vpn_mask |= ct.str.startswith(prefix)

    if "super_class" in fafb.neurons.columns:
        sc = fafb.neurons["super_class"].fillna("")
        vpn_mask |= sc.str.contains("visual_projection", case=False)

    return fafb.neurons[vpn_mask].copy()


def identify_motor_neurons(manc: ConnectomeData) -> pd.DataFrame:
    """Identify motor neurons in MANC.

    These are the final output layer of the network.
    Groups by effector:
    - Wing MNs: flight muscles (indirect & direct flight muscles)
    - Leg MNs: T1/T2/T3 leg neuropils
    - Haltere MNs: gyroscopic sensing / balance
    - Neck MNs: head positioning
    """
    mn_mask = pd.Series(False, index=manc.neurons.index)

    if "cell_type" in manc.neurons.columns:
        ct = manc.neurons["cell_type"].fillna("")
        mn_mask |= ct.str.startswith("MN")
        mn_mask |= ct.str.contains("motor", case=False)

    if "super_class" in manc.neurons.columns:
        sc = manc.neurons["super_class"].fillna("")
        mn_mask |= (sc == "motor_neuron") | sc.str.contains("motor", case=False)

    return manc.neurons[mn_mask].copy()


def identify_sensory_neurons(manc: ConnectomeData) -> pd.DataFrame:
    """Identify sensory neurons in MANC.

    Proprioceptive and mechanosensory neurons that provide input to the VNC.
    These form part of the input layer alongside photoreceptors.
    """
    sn_mask = pd.Series(False, index=manc.neurons.index)

    if "cell_type" in manc.neurons.columns:
        ct = manc.neurons["cell_type"].fillna("")
        sn_mask |= ct.str.startswith("SN") | ct.str.startswith("SA")

    if "super_class" in manc.neurons.columns:
        sc = manc.neurons["super_class"].fillna("")
        sn_mask |= sc.str.contains("sensory", case=False)

    return manc.neurons[sn_mask].copy()


def identify_photoreceptors(optic_lobe: ConnectomeData) -> pd.DataFrame:
    """Identify photoreceptor neurons (R1-R8) in the optic lobe.

    These are the primary visual input layer.
    - R1-R6: broadband, motion detection (project to lamina)
    - R7: UV-sensitive (project to medulla M6)
    - R8: blue/green-sensitive (project to medulla M3)
    """
    pr_mask = pd.Series(False, index=optic_lobe.neurons.index)

    if "cell_type" in optic_lobe.neurons.columns:
        ct = optic_lobe.neurons["cell_type"].fillna("")
        pr_mask |= ct.str.match(r"^R[1-8]")

    return optic_lobe.neurons[pr_mask].copy()


def match_cell_types_across_datasets(
    dataset_a: ConnectomeData,
    dataset_b: ConnectomeData,
) -> pd.DataFrame:
    """Match neurons between two datasets by cell type name.

    Cell type nomenclature is shared across FAFB, MANC, and optic lobe
    datasets thanks to systematic cross-validation efforts
    (Berg et al. 2025, Schlegel et al. 2024).

    Returns DataFrame with columns: id_a, id_b, cell_type, dataset_a, dataset_b
    """
    if "cell_type" not in dataset_a.neurons.columns:
        return pd.DataFrame()
    if "cell_type" not in dataset_b.neurons.columns:
        return pd.DataFrame()

    a = dataset_a.neurons[["id", "cell_type"]].dropna(subset=["cell_type"])
    b = dataset_b.neurons[["id", "cell_type"]].dropna(subset=["cell_type"])

    # Find shared cell types
    shared = set(a["cell_type"]) & set(b["cell_type"])

    if not shared:
        return pd.DataFrame()

    a_shared = a[a["cell_type"].isin(shared)]
    b_shared = b[b["cell_type"].isin(shared)]

    # For each shared cell type, create pairs
    matches = a_shared.merge(b_shared, on="cell_type", suffixes=("_a", "_b"))
    matches = matches.rename(columns={"id_a": "id_a", "id_b": "id_b"})
    matches["dataset_a"] = dataset_a.source_dataset
    matches["dataset_b"] = dataset_b.source_dataset

    return matches[["id_a", "id_b", "cell_type", "dataset_a", "dataset_b"]]


def classify_network_role(neurons: pd.DataFrame) -> pd.Series:
    """Assign each neuron a functional role for the trainable network.

    Roles:
    - input_visual: photoreceptors R1-R8
    - input_sensory: proprioceptive/mechanosensory in VNC
    - processing_visual: optic lobe interneurons
    - processing_central: central brain interneurons
    - processing_vnc: VNC interneurons
    - bridge_dn: descending neurons (brain → VNC)
    - bridge_an: ascending neurons (VNC → brain)
    - bridge_vpn: visual projection neurons (OL → central brain)
    - output_motor: motor neurons
    """
    role = pd.Series("processing", index=neurons.index)
    ct = neurons.get("cell_type", pd.Series("", index=neurons.index)).fillna("")
    sc = neurons.get("super_class", pd.Series("", index=neurons.index)).fillna("")
    src = neurons.get("source", pd.Series("", index=neurons.index)).fillna("")

    # Inputs
    role[ct.str.match(r"^R[1-8]")] = "input_visual"
    role[sc.str.contains("sensory", case=False)] = "input_sensory"

    # Outputs
    role[ct.str.startswith("MN") | sc.str.contains("motor", case=False)] = "output_motor"

    # Bridge neurons
    role[ct.str.startswith("DN") | sc.str.contains("descending", case=False)] = "bridge_dn"
    role[ct.str.startswith("AN") | sc.str.contains("ascending", case=False)] = "bridge_an"

    vpn_prefixes = ["LC", "LPLC", "LT", "ML", "VCH", "DCH", "VS", "HS"]
    for p in vpn_prefixes:
        role[ct.str.startswith(p)] = "bridge_vpn"

    # Processing regions
    role[(role == "processing") & (src == "optic_lobe")] = "processing_visual"
    role[(role == "processing") & (src == "fafb")] = "processing_central"
    role[(role == "processing") & (src == "manc")] = "processing_vnc"

    return role
