"""Differentiable neuron dynamics models.

Two models are provided:

1. RateNeuron: continuous rate-based model (simpler, fully differentiable).
   Each neuron has a membrane potential v that evolves as:
       tau * dv/dt = -v + W @ f(v) + I_ext
   where f is a nonlinearity (e.g. softplus) and W is the synapse matrix.

2. LIFNeuron: leaky integrate-and-fire with surrogate gradient.
   Uses the straight-through estimator or sigmoid surrogate for
   backpropagation through spikes.

Both enforce Dale's law: each neuron's output is multiplied by its
sign mask (+1 excitatory, -1 inhibitory) so the synapse weight matrix
can be parameterized as all-positive and the sign is applied separately.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RateNeuron(nn.Module):
    """Continuous rate-based neuron dynamics.

    The state is the membrane potential v ∈ R^n_neurons.
    Output firing rate: r = activation(v)

    Update rule (Euler discretization):
        v[t+1] = (1 - dt/tau) * v[t] + (dt/tau) * (W @ r[t] + I_ext[t] + bias)

    Parameters are:
    - tau: per-neuron time constants (learnable)
    - bias: per-neuron resting potentials (learnable)
    - The weight matrix W is handled externally in the full model.
    """

    def __init__(
        self,
        n_neurons: int,
        dt: float = 0.5e-3,
        tau_init: float = 10e-3,
        tau_min: float = 1e-3,
        tau_max: float = 200e-3,
        activation: str = "softplus",
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Learnable per-neuron time constant (parameterized in log space)
        self._log_tau = nn.Parameter(
            torch.full((n_neurons,), fill_value=float(torch.tensor(tau_init).log()))
        )
        # Learnable per-neuron bias (resting potential)
        self.bias = nn.Parameter(torch.zeros(n_neurons))

        self.activation_fn = {
            "softplus": F.softplus,
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        }[activation]

    @property
    def tau(self) -> torch.Tensor:
        """Time constants clamped to [tau_min, tau_max]."""
        return self._log_tau.exp().clamp(self.tau_min, self.tau_max)

    def firing_rate(self, v: torch.Tensor) -> torch.Tensor:
        """Compute firing rate from membrane potential."""
        return self.activation_fn(v)

    def step(
        self,
        v: torch.Tensor,
        synaptic_input: torch.Tensor,
        external_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single Euler step of membrane dynamics.

        Parameters
        ----------
        v : (batch, n_neurons)
            Current membrane potentials.
        synaptic_input : (batch, n_neurons)
            W @ r[t] — weighted input from other neurons.
        external_input : (batch, n_neurons), optional
            External stimulus current.
        """
        tau = self.tau
        alpha = self.dt / tau  # (n_neurons,)

        total_input = synaptic_input + self.bias
        if external_input is not None:
            total_input = total_input + external_input

        # Euler: v_new = (1 - alpha) * v + alpha * total_input
        v_new = (1.0 - alpha) * v + alpha * total_input
        return v_new

    def init_state(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
        """Initialize membrane potentials to zero."""
        return torch.zeros(batch_size, self.n_neurons, device=device)


class SurrogateSpike(torch.autograd.Function):
    """Surrogate gradient for spiking nonlinearity.

    Forward: Heaviside step (spike if v > threshold)
    Backward: sigmoid surrogate gradient
    """

    @staticmethod
    def forward(ctx, v, threshold, beta):
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        ctx.beta = beta
        return (v >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        sigmoid = torch.sigmoid(ctx.beta * (v - ctx.threshold))
        grad = grad_output * sigmoid * (1 - sigmoid) * ctx.beta
        return grad, None, None


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with surrogate gradients.

    Update rule:
        v[t+1] = (1 - dt/tau) * v[t] * (1 - spike[t]) + (dt/tau) * (I_syn + I_ext + bias)
        spike[t+1] = Heaviside(v[t+1] - threshold)

    The reset is multiplicative: after spiking, v is multiplied by (1 - spike),
    effectively resetting to 0. This is differentiable through the surrogate.
    """

    def __init__(
        self,
        n_neurons: int,
        dt: float = 0.5e-3,
        tau_init: float = 10e-3,
        threshold: float = 1.0,
        surrogate_beta: float = 10.0,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.dt = dt
        self.threshold = threshold
        self.surrogate_beta = surrogate_beta

        self._log_tau = nn.Parameter(
            torch.full((n_neurons,), fill_value=float(torch.tensor(tau_init).log()))
        )
        self.bias = nn.Parameter(torch.zeros(n_neurons))

    @property
    def tau(self) -> torch.Tensor:
        return self._log_tau.exp().clamp(1e-3, 200e-3)

    def step(
        self,
        v: torch.Tensor,
        spike: torch.Tensor,
        synaptic_input: torch.Tensor,
        external_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single LIF step.

        Returns (v_new, spike_new).
        """
        tau = self.tau
        alpha = self.dt / tau

        total_input = synaptic_input + self.bias
        if external_input is not None:
            total_input = total_input + external_input

        # Reset: multiply v by (1 - spike) to zero out after spike
        v_reset = v * (1.0 - spike)

        # Integrate
        v_new = (1.0 - alpha) * v_reset + alpha * total_input

        # Spike
        spike_new = SurrogateSpike.apply(v_new, self.threshold, self.surrogate_beta)

        return v_new, spike_new

    def init_state(
        self, batch_size: int, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize (v, spike) to zeros."""
        v = torch.zeros(batch_size, self.n_neurons, device=device)
        s = torch.zeros(batch_size, self.n_neurons, device=device)
        return v, s
