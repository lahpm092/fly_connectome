"""Configuration and data source URLs for fly connectome datasets."""

from pathlib import Path

# ── Default data directory ──────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2].parent / "data"

# ── FAFB (Female Adult Fly Brain) ──────────────────────────────────
# FlyWire whole-brain connectome: ~139K neurons, ~50M synapses
# Source: Dorkenwald et al., Nature 2024
FAFB_ZENODO_RECORD = "10676866"
FAFB_SYNAPSE_URL = (
    "https://zenodo.org/records/10676866/files/"
    "fafb_synapses_cleft_score_gt50.feather"
)
FAFB_EDGES_URL = (
    "https://zenodo.org/records/10676866/files/"
    "fafb_edges_by_neuropil.feather"
)
FAFB_CODEX_DOWNLOAD = "https://codex.flywire.ai/api/download?dataset=fafb"

# ── MANC (Male Adult Nerve Cord) ───────────────────────────────────
# ~23K neurons, ~10M presynaptic sites, ~74M postsynaptic densities
# Source: Takemura et al., eLife 2024
MANC_NEUPRINT_SERVER = "https://neuprint.janelia.org"
MANC_NEUPRINT_DATASET = "manc:v1.2.1"
MANC_FLAT_FILES_BUCKET = "gs://flyem-manc-exports/v1.0"

# ── Male Optic Lobe ───────────────────────────────────────────────
# Right optic lobe: ~50K neurons, 700+ cell types
# Source: Nern et al., Nature 2025
OPTIC_LOBE_NEUPRINT_SERVER = "https://neuprint.janelia.org"
OPTIC_LOBE_NEUPRINT_DATASET = "optic-lobe:v1.0"
OPTIC_LOBE_BUCKET = "gs://flyem-optic-lobe/v1.0"

# ── Male CNS (full, includes optic lobe + central brain + VNC) ────
# Source: Janelia FlyEM, 2025
MALE_CNS_NEUPRINT_SERVER = "https://neuprint.janelia.org"
MALE_CNS_NEUPRINT_DATASET = "cns:v1.0"

# ── Frankenbrain (pre-fused FAFB + MANC via BANC neck bridge) ─────
# Source: BANC project (Harvard), Lee et al.
# This SQLite DB already integrates FAFB brain + MANC VNC
FRANKENBRAIN_TABLES = {
    "meta": "Neuron metadata with cell types, regions, NT predictions",
    "edgelist_simple": "Neuron-to-neuron edges (FAFB + MANC unified)",
    "edgelist": "Compartment-level edges",
    "neck_bridge": "BANC neurons matching FAFB and MANC counterparts",
    "an_dn_edgelist_simple": "Ascending/descending neuron edges",
}

# ── Cross-dataset annotations ─────────────────────────────────────
OL_ANNOTATIONS_REPO = "https://github.com/flyconnectome/ol_annotations"
FLYWIRE_ANNOTATIONS_REPO = "https://github.com/flyconnectome/flywire_annotations"
NECK_CONNECTIVE_REPO = "https://github.com/flyconnectome/2023neckconnective"

# ── Neurotransmitter types ────────────────────────────────────────
# Used for Dale's law: constraining synapse sign
NEUROTRANSMITTERS = {
    "acetylcholine": "excitatory",
    "glutamate": "mixed",       # excitatory at NMJ, inhibitory in CNS
    "gaba": "inhibitory",
    "octopamine": "modulatory",
    "serotonin": "modulatory",
    "dopamine": "modulatory",
}

# ── Network architecture defaults ─────────────────────────────────
DEFAULT_DT = 0.5e-3          # 0.5 ms simulation timestep
DEFAULT_TAU = 10e-3           # 10 ms membrane time constant
SYNAPSE_COUNT_THRESHOLD = 5   # min synapses to count as connected (Codex default)
