"""Full trainable connectome neural network.

This module defines ConnectomeNetwork, a recurrent neural network whose
architecture is dictated by biological connectome data. The key innovation
(following Lappalainen et al. 2024, "flyvis") is that the connectivity
structure is fixed by biology while the synapse strengths and neuron
dynamics parameters are learned via gradient descent.

Architecture overview
=====================

    ┌──────────────────────────────────────────────────────────────────┐
    │                    FUSED CONNECTOME NETWORK                      │
    │                                                                  │
    │  Inputs:                                                         │
    │    ├─ Visual: photoreceptors R1-R8 (optic lobe)                 │
    │    └─ Sensory: proprioceptive/mechano (VNC)                     │
    │                                                                  │
    │  Processing layers (recurrent, connectome-constrained):          │
    │    ├─ Optic Lobe: Lamina → Medulla → Lobula/LP                  │
    │    ├─ Central Brain: MB, CX, LH, AVLP, ...                     │
    │    ├─ Bridge: VPNs (OL→CB), DNs (CB→VNC), ANs (VNC→CB)        │
    │    └─ VNC: premotor interneurons, CPGs                          │
    │                                                                  │
    │  Outputs:                                                        │
    │    └─ Motor neurons: wing, leg, haltere, neck MNs               │
    └──────────────────────────────────────────────────────────────────┘

The network unrolls in time using either:
  - Rate dynamics (continuous, RateNeuron)
  - Spiking dynamics (LIF with surrogate gradients)

Training strategy
=================
1. Initialize weights from synapse counts (log-scaled).
2. Enforce Dale's law via sign mask (weights = |W| * sign_mask).
3. Enforce sparsity: only connections present in the connectome are allowed.
4. Train with BPTT for supervised tasks or with RL (policy gradient) for
   sensorimotor control.

Weight parameterization
=======================
The raw weight matrix is stored as a dense parameter only at nonzero
positions of the connectome adjacency (sparse parameterization).
The effective weight matrix is:
    W_eff[i,j] = softplus(w_raw[i,j]) * sign_mask[i] * connectivity_mask[i,j]
This ensures:
    - Weights are non-negative (softplus)
    - Dale's law is enforced (sign_mask)
    - Sparsity is preserved (connectivity_mask)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from fly_connectome.network.graph import FusedConnectomeGraph
from fly_connectome.network.neuron import LIFNeuron, RateNeuron


class SparseWeights(nn.Module):
    """Learnable sparse weight matrix with Dale's law.

    Only stores parameters for edges that exist in the connectome.
    Reconstructs the full sparse matrix for forward pass.
    """

    def __init__(
        self,
        graph: FusedConnectomeGraph,
        device: str = "cpu",
    ):
        super().__init__()
        n = graph.n_neurons
        coo = graph.adjacency.tocoo()

        # Store topology (fixed, not learnable)
        self.register_buffer(
            "edge_indices",
            torch.tensor(
                np.vstack([coo.row, coo.col]), dtype=torch.long, device=device
            ),
        )
        self.n_neurons = n
        self.n_edges = len(coo.data)

        # Learnable raw weights (one per edge), initialized from connectome
        init_weights = torch.tensor(coo.data, dtype=torch.float32, device=device)
        self.raw_weights = nn.Parameter(init_weights)

        # Sign mask per neuron (fixed)
        self.register_buffer(
            "sign_mask",
            torch.tensor(graph.sign_mask, dtype=torch.float32, device=device),
        )

    def effective_weights(self) -> torch.Tensor:
        """Compute effective edge weights: softplus(raw) * sign[pre_neuron].

        Returns tensor of shape (n_edges,).
        """
        # Positive weights via softplus
        pos_weights = F.softplus(self.raw_weights)
        # Apply Dale's law: multiply by pre-neuron sign
        pre_signs = self.sign_mask[self.edge_indices[0]]
        # For unconstrained neurons (sign=0), allow free sign
        signs = torch.where(pre_signs == 0, torch.ones_like(pre_signs), pre_signs)
        return pos_weights * signs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse matrix-vector product: W @ x.

        x : (batch, n_neurons)
        returns : (batch, n_neurons)
        """
        w = self.effective_weights()
        # Build sparse matrix
        W_sparse = torch.sparse_coo_tensor(
            self.edge_indices, w, size=(self.n_neurons, self.n_neurons)
        )
        # (batch, n) @ (n, n)^T → need to transpose the product
        # torch.sparse.mm: (sparse, dense) → dense
        # We want y = x @ W^T = (W @ x^T)^T
        return torch.sparse.mm(W_sparse, x.t()).t()


class ConnectomeNetwork(nn.Module):
    """Full connectome-constrained recurrent neural network.

    This is the main model class. It takes a FusedConnectomeGraph and
    creates a trainable RNN with:
    - Connectome-constrained sparse connectivity
    - Dale's law enforcement
    - Per-neuron learnable dynamics (tau, bias)
    - Configurable neuron model (rate-based or LIF)

    Usage::

        graph = build_fused_graph(neurons, edges, roles)
        model = ConnectomeNetwork(graph, neuron_type="rate")

        # Forward pass: simulate T timesteps
        visual_input = torch.randn(batch, T, n_photoreceptors)
        sensory_input = torch.randn(batch, T, n_sensory)
        motor_output = model(visual_input, sensory_input, n_steps=T)
        # motor_output: (batch, T, n_motor_neurons)
    """

    def __init__(
        self,
        graph: FusedConnectomeGraph,
        neuron_type: str = "rate",
        dt: float = 0.5e-3,
        tau_init: float = 10e-3,
        device: str = "cpu",
    ):
        super().__init__()
        self.graph = graph
        self.n_neurons = graph.n_neurons
        self.device = device
        self.neuron_type = neuron_type

        # Sparse weight matrix
        self.weights = SparseWeights(graph, device=device)

        # Neuron dynamics
        if neuron_type == "rate":
            self.neurons = RateNeuron(
                graph.n_neurons, dt=dt, tau_init=tau_init
            ).to(device)
        elif neuron_type == "lif":
            self.neurons = LIFNeuron(
                graph.n_neurons, dt=dt, tau_init=tau_init
            ).to(device)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

        # Input projection: map external inputs to input neuron indices
        self.register_buffer(
            "input_indices",
            torch.tensor(graph.input_indices, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "output_indices",
            torch.tensor(graph.output_indices, dtype=torch.long, device=device),
        )

        self.n_inputs = len(graph.input_indices)
        self.n_outputs = len(graph.output_indices)

    def _inject_input(
        self,
        external: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Scatter external input into the full neuron vector.

        external : (batch, n_inputs)
        returns : (batch, n_neurons) with input at input_indices
        """
        full = torch.zeros(batch_size, self.n_neurons, device=external.device)
        full[:, self.input_indices] = external
        return full

    def _extract_output(self, state: torch.Tensor) -> torch.Tensor:
        """Extract motor neuron activity from full state.

        state : (batch, n_neurons)
        returns : (batch, n_outputs)
        """
        return state[:, self.output_indices]

    def forward(
        self,
        inputs: torch.Tensor,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """Simulate the network for T timesteps.

        Parameters
        ----------
        inputs : (batch, T, n_inputs)
            Time series of external inputs (visual + sensory concatenated).
        n_steps : int, optional
            Number of simulation steps. Defaults to inputs.shape[1].

        Returns
        -------
        outputs : (batch, T, n_outputs)
            Motor neuron activity over time.
        """
        batch_size, T, _ = inputs.shape
        if n_steps is None:
            n_steps = T

        outputs = []

        if self.neuron_type == "rate":
            v = self.neurons.init_state(batch_size, device=inputs.device)

            for t in range(n_steps):
                # Compute firing rates
                r = self.neurons.firing_rate(v)
                # Synaptic input: W @ r
                syn_input = self.weights(r)
                # External input
                ext = self._inject_input(
                    inputs[:, min(t, T - 1)], batch_size
                )
                # Update dynamics
                v = self.neurons.step(v, syn_input, ext)
                # Record output
                outputs.append(self._extract_output(r))

        elif self.neuron_type == "lif":
            v, spike = self.neurons.init_state(batch_size, device=inputs.device)

            for t in range(n_steps):
                # Synaptic input from spikes
                syn_input = self.weights(spike)
                # External input
                ext = self._inject_input(
                    inputs[:, min(t, T - 1)], batch_size
                )
                # Update LIF dynamics
                v, spike = self.neurons.step(v, spike, syn_input, ext)
                # Record output (spike rates of motor neurons)
                outputs.append(self._extract_output(spike))

        return torch.stack(outputs, dim=1)  # (batch, T, n_outputs)

    def parameter_summary(self) -> str:
        """Summarize learnable parameters."""
        lines = [
            f"=== ConnectomeNetwork ({self.neuron_type}) ===",
            f"  Total neurons:     {self.n_neurons:,}",
            f"  Synapse parameters: {self.weights.n_edges:,}",
            f"  Neuron tau params:  {self.n_neurons:,}",
            f"  Neuron bias params: {self.n_neurons:,}",
            f"  Input neurons:     {self.n_inputs:,}",
            f"  Output neurons:    {self.n_outputs:,}",
            f"  Total parameters:  {sum(p.numel() for p in self.parameters()):,}",
        ]
        return "\n".join(lines)
