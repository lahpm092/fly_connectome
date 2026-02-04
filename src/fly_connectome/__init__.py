"""Fused Drosophila connectome as a trainable neural network.

Combines three connectome datasets:
- FAFB: Female Adult Fly Brain (~139K neurons, ~50M synapses)
- MANC: Male Adult Nerve Cord (~23K neurons, ~74M postsynaptic densities)
- Optic Lobe: Male right optic lobe (~50K neurons, 700+ types)

Linked via descending/ascending neurons (neck connective) and
visual projection neurons (optic lobe -> central brain).
"""

__version__ = "0.1.0"
