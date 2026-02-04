"""Dataset-specific loaders that produce standardized DataFrames.

Each loader returns two DataFrames:
  neurons_df: neuron metadata (id, cell_type, region, neurotransmitter, ...)
  edges_df:   directed edges (pre_id, post_id, syn_count, nt_type, weight, ...)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ConnectomeData:
    """Standardized connectome dataset."""

    name: str
    neurons: pd.DataFrame   # columns: id, cell_type, super_class, region, nt_type, ...
    edges: pd.DataFrame      # columns: pre_id, post_id, syn_count, nt_type, weight
    source_dataset: str      # e.g. "fafb", "manc", "optic_lobe"

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def density(self) -> float:
        n = self.n_neurons
        return self.n_edges / (n * (n - 1)) if n > 1 else 0.0

    def summary(self) -> str:
        nt_dist = self.neurons["nt_type"].value_counts() if "nt_type" in self.neurons.columns else {}
        region_dist = self.neurons["region"].value_counts() if "region" in self.neurons.columns else {}
        lines = [
            f"=== {self.name} ({self.source_dataset}) ===",
            f"  Neurons:     {self.n_neurons:,}",
            f"  Edges:       {self.n_edges:,}",
            f"  Density:     {self.density:.6f}",
            f"  Avg degree:  {self.n_edges / max(self.n_neurons, 1):.1f}",
        ]
        if len(nt_dist):
            lines.append("  Neurotransmitters:")
            for nt, cnt in nt_dist.head(6).items():
                lines.append(f"    {nt}: {cnt:,}")
        if len(region_dist):
            lines.append(f"  Brain regions: {len(region_dist)} unique")
            for reg, cnt in region_dist.head(8).items():
                lines.append(f"    {reg}: {cnt:,}")
        return "\n".join(lines)


# ── FAFB loader ─────────────────────────────────────────────────────

def load_fafb_from_feather(data_dir: Path) -> ConnectomeData:
    """Load FAFB from Zenodo feather files."""
    edges_path = data_dir / "edges_by_neuropil.feather"
    syn_path = data_dir / "synapses.feather"

    if edges_path.exists():
        edges_raw = pd.read_feather(edges_path)
    else:
        raise FileNotFoundError(f"FAFB edges not found at {edges_path}")

    # Standardize column names
    col_map = {
        "pre_pt_root_id": "pre_id",
        "post_pt_root_id": "post_id",
        "syn_count": "syn_count",
        "neuropil": "region",
        "nt_type": "nt_type",
    }
    edges = edges_raw.rename(columns={
        k: v for k, v in col_map.items() if k in edges_raw.columns
    })

    # Aggregate edges per neuron pair (sum across neuropils)
    if "region" in edges.columns:
        edges_agg = (
            edges.groupby(["pre_id", "post_id"], as_index=False)
            .agg(syn_count=("syn_count", "sum"))
        )
    else:
        edges_agg = edges[["pre_id", "post_id", "syn_count"]].copy()

    # Build neuron table from unique IDs in edge list
    all_ids = pd.unique(np.concatenate([
        edges_agg["pre_id"].values, edges_agg["post_id"].values
    ]))
    neurons = pd.DataFrame({"id": all_ids})
    neurons["source"] = "fafb"

    # If per-neuropil data has NT info, get dominant NT per pre-neuron
    if "nt_type" in edges.columns:
        dominant_nt = (
            edges.groupby("pre_id")["nt_type"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else "unknown")
            .reset_index()
            .rename(columns={"pre_id": "id", "nt_type": "nt_type"})
        )
        neurons = neurons.merge(dominant_nt, on="id", how="left")

    # Compute normalized weight from synapse count
    edges_agg["weight"] = edges_agg["syn_count"].astype(float)

    return ConnectomeData(
        name="FAFB (Female Adult Fly Brain)",
        neurons=neurons,
        edges=edges_agg,
        source_dataset="fafb",
    )


# ── MANC loader ─────────────────────────────────────────────────────

def load_manc_from_neuprint(
    server: str = "neuprint.janelia.org",
    dataset: str = "manc:v1.2.1",
    token: str | None = None,
    min_syn: int = 5,
) -> ConnectomeData:
    """Load MANC dataset via neuPrint API.

    Parameters
    ----------
    min_syn : int
        Minimum synapse count to include an edge (default 5, matching Codex).
    """
    from neuprint import Client, NeuronCriteria as NC

    client = Client(server, dataset=dataset, token=token)

    # Fetch all traced neurons
    neurons_raw, roi_counts = client.fetch_neurons(NC(status="Traced"))
    neurons = pd.DataFrame({
        "id": neurons_raw["bodyId"],
        "cell_type": neurons_raw.get("type", pd.Series(dtype=str)),
        "instance": neurons_raw.get("instance", pd.Series(dtype=str)),
        "region": neurons_raw.get("roiInfo", pd.Series(dtype=str)),
        "nt_type": neurons_raw.get("predictedNt", pd.Series(dtype=str)),
        "source": "manc",
    })

    # Fetch adjacency matrix
    adj_df = client.fetch_adjacencies(NC(status="Traced"), NC(status="Traced"))
    edges = pd.DataFrame({
        "pre_id": adj_df["bodyId_pre"],
        "post_id": adj_df["bodyId_post"],
        "syn_count": adj_df["weight"],
    })
    edges = edges[edges["syn_count"] >= min_syn].reset_index(drop=True)
    edges["weight"] = edges["syn_count"].astype(float)

    return ConnectomeData(
        name="MANC (Male Adult Nerve Cord)",
        neurons=neurons,
        edges=edges,
        source_dataset="manc",
    )


# ── Optic Lobe loader ──────────────────────────────────────────────

def load_optic_lobe_from_neuprint(
    server: str = "neuprint.janelia.org",
    dataset: str = "optic-lobe:v1.0",
    token: str | None = None,
    min_syn: int = 5,
) -> ConnectomeData:
    """Load male right optic lobe via neuPrint API."""
    from neuprint import Client, NeuronCriteria as NC

    client = Client(server, dataset=dataset, token=token)

    neurons_raw, roi_counts = client.fetch_neurons(NC(status="Traced"))
    neurons = pd.DataFrame({
        "id": neurons_raw["bodyId"],
        "cell_type": neurons_raw.get("type", pd.Series(dtype=str)),
        "instance": neurons_raw.get("instance", pd.Series(dtype=str)),
        "region": neurons_raw.get("roiInfo", pd.Series(dtype=str)),
        "nt_type": neurons_raw.get("predictedNt", pd.Series(dtype=str)),
        "source": "optic_lobe",
    })

    adj_df = client.fetch_adjacencies(NC(status="Traced"), NC(status="Traced"))
    edges = pd.DataFrame({
        "pre_id": adj_df["bodyId_pre"],
        "post_id": adj_df["bodyId_post"],
        "syn_count": adj_df["weight"],
    })
    edges = edges[edges["syn_count"] >= min_syn].reset_index(drop=True)
    edges["weight"] = edges["syn_count"].astype(float)

    return ConnectomeData(
        name="Male Optic Lobe (Right)",
        neurons=neurons,
        edges=edges,
        source_dataset="optic_lobe",
    )


# ── Frankenbrain loader (pre-fused FAFB + MANC) ────────────────────

def load_frankenbrain(sqlite_path: Path) -> ConnectomeData:
    """Load the pre-fused FAFB+MANC connectome from BANC frankenbrain SQLite.

    The frankenbrain_v1.1_data.sqlite already contains the integrated
    FAFB brain + MANC VNC connected via BANC neck bridge neurons.
    """
    conn = sqlite3.connect(str(sqlite_path))

    # Load metadata
    neurons = pd.read_sql("SELECT * FROM meta", conn)
    neurons = neurons.rename(columns={
        "root_id": "id",
        "cell_type": "cell_type",
        "super_class": "super_class",
        "nt_type": "nt_type",
    })

    # Load edge list
    edges = pd.read_sql("SELECT * FROM edgelist_simple", conn)
    edges = edges.rename(columns={
        "pre_root_id": "pre_id",
        "post_root_id": "post_id",
        "syn_count": "syn_count",
    })
    if "syn_count" in edges.columns:
        edges["weight"] = edges["syn_count"].astype(float)

    # Load neck bridge for reference
    try:
        bridge = pd.read_sql("SELECT * FROM neck_bridge", conn)
    except Exception:
        bridge = pd.DataFrame()

    conn.close()

    data = ConnectomeData(
        name="Frankenbrain (FAFB + MANC via BANC bridge)",
        neurons=neurons,
        edges=edges,
        source_dataset="frankenbrain",
    )
    data.neck_bridge = bridge
    return data
