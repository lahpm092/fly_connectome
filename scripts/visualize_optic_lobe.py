#!/usr/bin/env python3
"""Visualize the optic lobe subset of the fused connectome.

Shows the layered visual processing circuit from photoreceptors
through lamina, medulla, lobula/LP to visual projection neurons.
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from scripts.analyze_and_fuse import create_demo_datasets
from fly_connectome.data.bridge import classify_network_role
from fly_connectome.network.graph import build_fused_graph


def extract_optic_lobe_subgraph(neurons, edges):
    """Extract the optic lobe neurons and their immediate downstream targets."""

    # Optic lobe neurons
    ol_mask = neurons["source"] == "optic_lobe"

    # Also grab VPN targets in central brain (first hop)
    vpn_prefixes = ("LC", "LPLC", "LT", "VS", "HS")
    vpn_ids = set(
        neurons.loc[
            neurons["cell_type"].fillna("").str.startswith(vpn_prefixes), "id"
        ]
    )

    # Find central brain neurons directly postsynaptic to VPNs
    vpn_targets = set(
        edges.loc[edges["pre_id"].isin(vpn_ids), "post_id"]
    )
    cb_first_hop = neurons[neurons["id"].isin(vpn_targets) & (neurons["source"] == "fafb")]

    # Combine
    ol_neurons = pd.concat([neurons[ol_mask], cb_first_hop]).drop_duplicates(subset="id")
    ol_ids = set(ol_neurons["id"])

    # Edges within this subgraph
    ol_edges = edges[edges["pre_id"].isin(ol_ids) & edges["post_id"].isin(ol_ids)].copy()

    return ol_neurons, ol_edges


def assign_layer(cell_type):
    """Map cell type to its neuropil layer in the optic lobe."""
    ct = str(cell_type)
    if ct.startswith("R"):
        return "0_Retina"
    # Check multi-char prefixes BEFORE single-char "L"
    if ct.startswith(("LC", "LPLC", "LT")):
        return "5_VPN (OL→Brain)"
    if ct.startswith(("LH", "AVLP", "SLP")):
        return "6_Central Brain"
    if ct.startswith("L"):
        return "1_Lamina"
    if ct.startswith("Mi"):
        return "2_Medulla"
    if ct.startswith("Tm"):
        return "3_Medulla→Lobula"
    if ct.startswith(("T4", "T5")):
        return "4_Lobula Plate"
    if ct.startswith(("VS", "HS")):
        return "5_VPN (OL→Brain)"
    if ct == "OL_inter":
        return "3_Medulla→Lobula"
    return "3_Medulla→Lobula"


# ── Color scheme ────────────────────────────────────────────────
LAYER_COLORS = {
    "0_Retina":           "#e41a1c",   # red
    "1_Lamina":           "#ff7f00",   # orange
    "2_Medulla":          "#ffd700",   # gold
    "3_Medulla→Lobula":   "#4daf4a",   # green
    "4_Lobula Plate":     "#377eb8",   # blue
    "5_VPN (OL→Brain)":   "#984ea3",   # purple
    "6_Central Brain":    "#a65628",   # brown
}

NT_MARKERS = {
    "acetylcholine": "o",    # circle = excitatory
    "glutamate":     "s",    # square = inhibitory (in CNS)
    "gaba":          "D",    # diamond = inhibitory
    "dopamine":      "^",    # triangle = modulatory
    "unknown":       "o",
}


def print_connectivity_table(ol_neurons, ol_edges):
    """Print a detailed text-based connectivity matrix by layer."""

    ol_neurons = ol_neurons.copy()
    ol_neurons["layer"] = ol_neurons["cell_type"].apply(assign_layer)

    # Aggregate edges by cell type
    id_to_type = dict(zip(ol_neurons["id"], ol_neurons["cell_type"]))
    id_to_layer = dict(zip(ol_neurons["id"], ol_neurons["layer"]))

    type_edges = defaultdict(lambda: {"count": 0, "total_syn": 0})
    for _, row in ol_edges.iterrows():
        pre_type = id_to_type.get(row["pre_id"], "?")
        post_type = id_to_type.get(row["post_id"], "?")
        key = (pre_type, post_type)
        type_edges[key]["count"] += 1
        type_edges[key]["total_syn"] += row.get("syn_count", row.get("weight", 1))

    # ── Print neuron inventory ──
    print("=" * 72)
    print("OPTIC LOBE SUBNETWORK — NEURON INVENTORY")
    print("=" * 72)

    for layer in sorted(LAYER_COLORS.keys()):
        layer_neurons = ol_neurons[ol_neurons["layer"] == layer]
        if len(layer_neurons) == 0:
            continue
        layer_name = layer.split("_", 1)[1]
        print(f"\n  ┌─ {layer_name} ─{'─' * (50 - len(layer_name))}┐")
        for ct, group in layer_neurons.groupby("cell_type"):
            nt = group["nt_type"].mode().iloc[0] if len(group["nt_type"].mode()) else "?"
            sign = {"acetylcholine": "+", "gaba": "−", "glutamate": "−",
                    "dopamine": "~"}.get(nt, "?")
            print(f"  │  {ct:15s}  n={len(group):3d}  NT={nt:15s} [{sign}]  │")
        print(f"  └─{'─' * 56}┘")

    # ── Print connectivity ──
    print("\n" + "=" * 72)
    print("OPTIC LOBE SUBNETWORK — CONNECTION TABLE")
    print("=" * 72)
    print(f"\n  {'Presynaptic':>18s}  →  {'Postsynaptic':18s}  "
          f"{'Edges':>6s}  {'Total Syn':>10s}  {'Avg Syn':>8s}")
    print("  " + "─" * 70)

    # Sort by layer order
    def layer_sort_key(ct):
        return assign_layer(ct)

    sorted_edges = sorted(
        type_edges.items(),
        key=lambda x: (layer_sort_key(x[0][0]), layer_sort_key(x[0][1]), x[0][0])
    )

    current_pre_layer = None
    for (pre_type, post_type), stats in sorted_edges:
        pre_layer = assign_layer(pre_type).split("_", 1)[1]
        if pre_layer != current_pre_layer:
            current_pre_layer = pre_layer
            print(f"\n  ── {pre_layer} ──")

        avg_syn = stats["total_syn"] / max(stats["count"], 1)
        # Arrow style based on direction
        pre_l = assign_layer(pre_type)
        post_l = assign_layer(post_type)
        if pre_l < post_l:
            arrow = "──▶"   # feedforward
        elif pre_l > post_l:
            arrow = "◀──"   # feedback
        else:
            arrow = "◀─▶"   # lateral
        print(f"  {pre_type:>18s}  {arrow}  {post_type:18s}  "
              f"{stats['count']:6d}  {stats['total_syn']:10.0f}  {avg_syn:8.1f}")

    # ── Summary stats ──
    print(f"\n  Total neurons in subgraph: {len(ol_neurons):,}")
    print(f"  Total edges in subgraph:  {len(ol_edges):,}")
    print(f"  Unique cell type pairs:   {len(type_edges):,}")


def plot_optic_lobe_graph(ol_neurons, ol_edges, out_path="optic_lobe_circuit.png"):
    """Generate a layered graph visualization of the optic lobe circuit."""

    ol_neurons = ol_neurons.copy()
    ol_neurons["layer"] = ol_neurons["cell_type"].apply(assign_layer)

    id_to_type = dict(zip(ol_neurons["id"], ol_neurons["cell_type"]))
    id_to_layer = dict(zip(ol_neurons["id"], ol_neurons["layer"]))

    # ── Collapse to cell-type graph (too many individual neurons) ──
    G = nx.DiGraph()

    # Add nodes (one per cell type)
    for ct, group in ol_neurons.groupby("cell_type"):
        layer = group["layer"].iloc[0]
        nt = group["nt_type"].mode().iloc[0] if len(group["nt_type"].mode()) else "unknown"
        G.add_node(ct, layer=layer, count=len(group), nt=nt)

    # Add edges (aggregated by type pair)
    type_edges = defaultdict(float)
    for _, row in ol_edges.iterrows():
        pre_t = id_to_type.get(row["pre_id"])
        post_t = id_to_type.get(row["post_id"])
        if pre_t and post_t and pre_t != post_t:
            type_edges[(pre_t, post_t)] += row.get("syn_count", 1)

    for (pre, post), syn in type_edges.items():
        if pre in G and post in G:
            G.add_edge(pre, post, weight=syn)

    # ── Layout: vertical layers ──
    layer_order = sorted(set(nx.get_node_attributes(G, "layer").values()))
    layer_y = {l: i * 2.0 for i, l in enumerate(reversed(layer_order))}

    # Spread nodes horizontally within each layer
    layer_nodes = defaultdict(list)
    for node in G.nodes():
        layer_nodes[G.nodes[node]["layer"]].append(node)

    pos = {}
    for layer, nodes in layer_nodes.items():
        nodes_sorted = sorted(nodes)
        n = len(nodes_sorted)
        x_spread = max(n * 1.2, 4)
        for i, node in enumerate(nodes_sorted):
            x = (i - n / 2) * (x_spread / max(n, 1))
            pos[node] = (x, layer_y[layer])

    # ── Draw ──
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Draw edges
    max_weight = max(d["weight"] for _, _, d in G.edges(data=True)) if G.edges() else 1
    for u, v, d in G.edges(data=True):
        weight = d["weight"]
        alpha = 0.15 + 0.6 * (weight / max_weight)
        width = 0.3 + 2.5 * (weight / max_weight)

        # Color edge by pre-synaptic NT
        nt = G.nodes[u].get("nt", "unknown")
        if nt == "acetylcholine":
            edge_color = "#66ff66"  # green = excitatory
        elif nt in ("gaba", "glutamate"):
            edge_color = "#ff6666"  # red = inhibitory
        else:
            edge_color = "#888888"

        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>",
                color=edge_color,
                alpha=alpha,
                lw=width,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    # Draw nodes
    for node in G.nodes():
        x, y = pos[node]
        layer = G.nodes[node]["layer"]
        count = G.nodes[node]["count"]
        nt = G.nodes[node]["nt"]
        color = LAYER_COLORS.get(layer, "#888888")
        size = 80 + count * 8

        marker = NT_MARKERS.get(nt, "o")
        ax.scatter(x, y, s=size, c=color, marker=marker,
                   edgecolors="white", linewidths=0.8, zorder=5)

        # Label
        fontsize = 7 if count < 15 else 8
        ax.text(x, y - 0.35, node, ha="center", va="top",
                fontsize=fontsize, color="white", fontweight="bold",
                zorder=6)

    # Layer labels
    for layer, y in layer_y.items():
        layer_name = layer.split("_", 1)[1]
        ax.text(-12, y, layer_name, ha="right", va="center",
                fontsize=11, color=LAYER_COLORS.get(layer, "white"),
                fontweight="bold")
        ax.axhline(y=y, color=LAYER_COLORS.get(layer, "#333"), alpha=0.15,
                   linestyle="--", zorder=0)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#66ff66", alpha=0.7, label="Excitatory (ACh)"),
        mpatches.Patch(facecolor="#ff6666", alpha=0.7, label="Inhibitory (GABA/Glu)"),
        mpatches.Patch(facecolor="#888888", alpha=0.7, label="Modulatory/Unknown"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=9, facecolor="#1a1a2e", edgecolor="#444",
              labelcolor="white")

    ax.set_xlim(-14, 14)
    ax.set_ylim(-1, max(layer_y.values()) + 1)
    ax.set_title("Drosophila Optic Lobe → Central Brain Circuit\n"
                 "(cell-type level connectivity, node size ∝ neuron count)",
                 fontsize=14, color="white", pad=20)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Plot saved to: {out_path}")


def print_signal_flow_trace(ol_neurons, ol_edges):
    """Trace a signal from a single photoreceptor through the network."""

    id_to_type = dict(zip(ol_neurons["id"], ol_neurons["cell_type"]))

    # Build adjacency by ID
    adj = defaultdict(list)
    for _, row in ol_edges.iterrows():
        adj[row["pre_id"]].append((row["post_id"], row.get("syn_count", 1)))

    # Pick one R1 photoreceptor
    r1_ids = ol_neurons[ol_neurons["cell_type"] == "R1"]["id"].values
    if len(r1_ids) == 0:
        print("  No R1 photoreceptors found.")
        return

    start = r1_ids[0]
    print("\n" + "=" * 72)
    print(f"SIGNAL TRACE from photoreceptor {start}")
    print("=" * 72)
    print(f"  Following strongest connections at each layer:\n")

    visited_types = set()
    current = start
    depth = 0
    max_depth = 8

    while depth < max_depth:
        ct = id_to_type.get(current, "?")
        layer = assign_layer(ct).split("_", 1)[1]
        nt = ol_neurons.loc[ol_neurons["id"] == current, "nt_type"].values
        nt_str = nt[0] if len(nt) else "?"

        indent = "  " + "  │  " * depth
        sign = {"acetylcholine": "+", "gaba": "−", "glutamate": "−"}.get(nt_str, "?")
        print(f"{indent}╔═ {ct} (id={current})")
        print(f"{indent}║  Layer: {layer}  |  NT: {nt_str} [{sign}]")

        # Find downstream targets
        targets = adj.get(current, [])
        if not targets:
            print(f"{indent}╚═ (terminal — no downstream targets)")
            break

        # Group by type
        type_syn = defaultdict(float)
        type_ids = defaultdict(list)
        for tid, syn in targets:
            tt = id_to_type.get(tid, "?")
            type_syn[tt] += syn
            type_ids[tt].append(tid)

        # Sort by total synapse weight
        sorted_targets = sorted(type_syn.items(), key=lambda x: -x[1])

        print(f"{indent}║  Downstream targets ({len(targets)} connections):")
        for tt, total_syn in sorted_targets[:5]:
            n_neurons = len(type_ids[tt])
            avg = total_syn / n_neurons
            marker = " ◀── NEXT" if tt == sorted_targets[0][0] and tt not in visited_types else ""
            print(f"{indent}║    → {tt:15s}  (n={n_neurons:2d}, "
                  f"total_syn={total_syn:6.0f}, avg={avg:5.1f}){marker}")
        if len(sorted_targets) > 5:
            print(f"{indent}║    ... and {len(sorted_targets) - 5} more types")

        print(f"{indent}╚═")

        # Follow strongest unvisited type
        visited_types.add(ct)
        next_type = None
        for tt, _ in sorted_targets:
            if tt not in visited_types:
                next_type = tt
                break
        if next_type is None:
            print(f"\n  {'  │  ' * (depth + 1)}(all downstream types already visited)")
            break

        # Pick first neuron of that type
        current = type_ids[next_type][0]
        depth += 1

    print(f"\n  Trace depth: {depth + 1} layers")


if __name__ == "__main__":
    # Generate the synthetic fused connectome
    print("Generating fused connectome...")
    neurons, edges = create_demo_datasets()

    # Extract optic lobe subgraph
    print("Extracting optic lobe subnetwork...\n")
    ol_neurons, ol_edges = extract_optic_lobe_subgraph(neurons, edges)

    # Print detailed tables
    print_connectivity_table(ol_neurons, ol_edges)

    # Signal flow trace
    print_signal_flow_trace(ol_neurons, ol_edges)

    # Generate plot
    print("\nGenerating circuit diagram...")
    plot_optic_lobe_graph(ol_neurons, ol_edges, "optic_lobe_circuit.png")
