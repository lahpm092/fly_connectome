"""Build a unified directed graph from the fused connectome.

This module takes the three (or pre-fused) connectome datasets and
produces a single sparse adjacency matrix suitable for constructing
a trainable PyTorch neural network.

Key design decisions
====================
1. Each biological neuron maps to exactly one node.
2. Each directed synapse bundle (pre→post, aggregated) maps to one edge.
3. Synapse weights are initialized from synapse counts (log-transformed).
4. Dale's law: each neuron's output sign is fixed by its neurotransmitter type.
   - ACh → excitatory (+)
   - GABA → inhibitory (−)
   - Glutamate → inhibitory in CNS (−), excitatory at NMJ (+)
   - Unknown → unconstrained
5. The graph is stored as a sparse COO tensor for efficient PyTorch operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


@dataclass
class FusedConnectomeGraph:
    """Unified connectome as a sparse directed graph ready for PyTorch.

    Attributes
    ----------
    n_neurons : int
        Total neuron count across all datasets.
    neuron_ids : np.ndarray
        Original neuron IDs, shape (n_neurons,).
    neuron_metadata : pd.DataFrame
        Full metadata for each neuron (indexed 0..n_neurons-1).
    adjacency : sp.coo_matrix
        Sparse adjacency matrix (n_neurons x n_neurons).
        Values are initial synapse weights (log-scaled counts).
    sign_mask : np.ndarray
        Per-neuron sign constraint: +1 (excitatory), -1 (inhibitory),
        0 (unconstrained). Shape (n_neurons,).
    role : np.ndarray
        Functional role string for each neuron. Shape (n_neurons,).
    input_indices : np.ndarray
        Indices of input neurons (photoreceptors + sensory).
    output_indices : np.ndarray
        Indices of output neurons (motor neurons).
    """

    n_neurons: int
    neuron_ids: np.ndarray
    neuron_metadata: pd.DataFrame
    adjacency: sp.coo_matrix
    sign_mask: np.ndarray
    role: np.ndarray
    input_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    output_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def to_torch_sparse(self, device: str = "cpu") -> torch.Tensor:
        """Convert adjacency to a PyTorch sparse COO tensor."""
        coo = self.adjacency.tocoo()
        indices = torch.tensor(
            np.vstack([coo.row, coo.col]), dtype=torch.long, device=device
        )
        values = torch.tensor(coo.data, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(
            indices, values, size=(self.n_neurons, self.n_neurons), device=device
        )

    def sign_mask_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Sign mask as a torch tensor."""
        return torch.tensor(self.sign_mask, dtype=torch.float32, device=device)

    def summary(self) -> str:
        roles, counts = np.unique(self.role, return_counts=True)
        lines = [
            "=== Fused Connectome Graph ===",
            f"  Total neurons:  {self.n_neurons:,}",
            f"  Total edges:    {self.adjacency.nnz:,}",
            f"  Input neurons:  {len(self.input_indices):,}",
            f"  Output neurons: {len(self.output_indices):,}",
            f"  Sparsity:       {1 - self.adjacency.nnz / (self.n_neurons**2):.6f}",
            "",
            "  Functional roles:",
        ]
        for r, c in sorted(zip(roles, counts), key=lambda x: -x[1]):
            lines.append(f"    {r}: {c:,}")

        signs, scounts = np.unique(self.sign_mask, return_counts=True)
        lines.append("")
        lines.append("  Sign constraints:")
        sign_labels = {1: "excitatory (+1)", -1: "inhibitory (-1)", 0: "unconstrained (0)"}
        for s, c in zip(signs, scounts):
            lines.append(f"    {sign_labels.get(int(s), str(s))}: {c:,}")

        return "\n".join(lines)


def _assign_sign(nt_type: str, role: str) -> int:
    """Assign Dale's law sign based on neurotransmitter type."""
    if pd.isna(nt_type) or nt_type == "" or nt_type == "unknown":
        return 0
    nt = nt_type.lower()
    if nt in ("acetylcholine", "ach"):
        return 1
    if nt == "gaba":
        return -1
    if nt in ("glutamate", "glu"):
        # Glutamate is inhibitory in CNS but excitatory at NMJ
        if role == "output_motor":
            return 1
        return -1
    # Modulatory NTs: treat as unconstrained
    return 0


def build_fused_graph(
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
    roles: pd.Series | None = None,
    min_syn: int = 5,
    log_scale_weights: bool = True,
) -> FusedConnectomeGraph:
    """Build a unified sparse graph from neuron + edge DataFrames.

    Parameters
    ----------
    neurons : pd.DataFrame
        Must have column 'id'. Optionally: 'nt_type', 'cell_type', 'source'.
    edges : pd.DataFrame
        Must have columns 'pre_id', 'post_id', 'syn_count' or 'weight'.
    roles : pd.Series, optional
        Functional role for each neuron (aligned with neurons index).
    min_syn : int
        Minimum synapse count threshold.
    log_scale_weights : bool
        If True, use log(1 + syn_count) as initial weight.
    """
    # Filter edges
    if "syn_count" in edges.columns:
        edges = edges[edges["syn_count"] >= min_syn].copy()

    # Build ID → index mapping
    all_ids = neurons["id"].values
    id_to_idx = {nid: i for i, nid in enumerate(all_ids)}
    n = len(all_ids)

    # Map edge IDs to indices
    valid_pre = edges["pre_id"].isin(id_to_idx)
    valid_post = edges["post_id"].isin(id_to_idx)
    edges = edges[valid_pre & valid_post].copy()

    row = edges["pre_id"].map(id_to_idx).values
    col = edges["post_id"].map(id_to_idx).values

    if "weight" in edges.columns:
        weights = edges["weight"].values.astype(np.float32)
    elif "syn_count" in edges.columns:
        weights = edges["syn_count"].values.astype(np.float32)
    else:
        weights = np.ones(len(edges), dtype=np.float32)

    if log_scale_weights:
        weights = np.log1p(weights)

    # Build sparse adjacency
    adj = sp.coo_matrix((weights, (row, col)), shape=(n, n))

    # Assign roles
    if roles is not None:
        role_arr = roles.values
    else:
        role_arr = np.array(["processing"] * n)

    # Sign mask (Dale's law)
    nt_types = neurons.get("nt_type", pd.Series("", index=neurons.index)).fillna("")
    sign_mask = np.array([
        _assign_sign(nt, r) for nt, r in zip(nt_types, role_arr)
    ], dtype=np.float32)

    # Identify input / output indices
    input_idx = np.where(np.isin(role_arr, ["input_visual", "input_sensory"]))[0]
    output_idx = np.where(role_arr == "output_motor")[0]

    return FusedConnectomeGraph(
        n_neurons=n,
        neuron_ids=all_ids,
        neuron_metadata=neurons.reset_index(drop=True),
        adjacency=adj,
        sign_mask=sign_mask,
        role=role_arr,
        input_indices=input_idx,
        output_indices=output_idx,
    )


def merge_connectome_datasets(
    *datasets,
    bridge_edges: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge multiple ConnectomeData objects into unified neuron + edge tables.

    Parameters
    ----------
    datasets : ConnectomeData
        Two or more connectome datasets.
    bridge_edges : pd.DataFrame, optional
        Additional edges connecting datasets (e.g. from neck bridge matching).
        Must have columns: pre_id, post_id, syn_count.

    Returns
    -------
    neurons : pd.DataFrame
    edges : pd.DataFrame
    """
    all_neurons = []
    all_edges = []

    for ds in datasets:
        n = ds.neurons.copy()
        n["source"] = ds.source_dataset
        all_neurons.append(n)
        all_edges.append(ds.edges.copy())

    neurons = pd.concat(all_neurons, ignore_index=True)

    # Check for ID collisions across datasets
    dup_ids = neurons.groupby("id")["source"].nunique()
    collisions = dup_ids[dup_ids > 1]
    if len(collisions):
        # Prefix IDs with source to disambiguate
        neurons["id"] = neurons["source"] + "_" + neurons["id"].astype(str)
        for i, ds in enumerate(datasets):
            all_edges[i]["pre_id"] = ds.source_dataset + "_" + all_edges[i]["pre_id"].astype(str)
            all_edges[i]["post_id"] = ds.source_dataset + "_" + all_edges[i]["post_id"].astype(str)
        if bridge_edges is not None:
            # Bridge edges reference original IDs — need source-qualified IDs
            pass  # caller must handle

    edges = pd.concat(all_edges, ignore_index=True)

    if bridge_edges is not None:
        edges = pd.concat([edges, bridge_edges], ignore_index=True)

    # Deduplicate edges (sum weights for same pre→post pair)
    if "syn_count" in edges.columns:
        edges = (
            edges.groupby(["pre_id", "post_id"], as_index=False)
            .agg(syn_count=("syn_count", "sum"))
        )
    edges["weight"] = edges.get("syn_count", edges.get("weight", 1)).astype(float)

    return neurons, edges
