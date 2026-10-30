# Square-FOVI

Square-FOVI explores an extended foveated interface for deep vision models. Building on the [FOVI](https://github.com/nblauch/fovi) architecture, it proposes a new sampling geometry and patching mechanism based on concentric square sampling and Chebyshev topologies, with the goal of making foveated Vision Transformers more hardware-efficient.

> **Status: exploratory, work in progress.** Most modules below are at the design stage. See [Implementation status](#implementation-status).

## Proposed architectural extensions

* **Concentric Square Node Sampling:** replaces radial sampling with an equal-area square geometry, snapping nodes to corners and edges.
* **DoB Receptive Fields & SAT Hypernetwork:** replaces the standard receptive field with a hypernetwork generating $k \times 3 \times 2$ parameters (using SiLU and tanh) for a Difference of Gaussians/Boxes (DoB) filter.
* **Chebyshev Manifold Patching:** uses Chebyshev ($L_\infty$) kNN on the manifold for native kernel correspondence, and bilinear interpolation for idiosyncratic spatial positions.
* **Log-Chebyshev P-RoPE:** custom positional embeddings replacing axial RoPE for DINOv3 backbones.
* **Semantic Fixation Points:** uses heavily downsampled early DINOv3 layers to extract semantic interest points for dynamic fixation.

## Implementation status

| Component | Status |
|---|---|
| Benchmarking and profiling tools | Extended from FOVI |
| Concentric square sampling | Prototype, not yet thoroughly tested |
| DoB receptive fields, Chebyshev patching, Log-Chebyshev P-RoPE, semantic fixation | Designed, not implemented |

## Installation

```bash
git clone https://github.com/arthursolere/Square-FOVI.git
cd Square-FOVI
pip install -e .
```

## Credits and acknowledgements

This project is a hard fork of the original [FOVI repository](https://github.com/nblauch/fovi).

* The core architecture and initial implementation were developed by **Nicholas Blauch**.
* The Square-FOVI extensions listed above were designed by **Arthur Solère (2026)**, who also extended the profiling tools and prototyped the square sampling.

## License

This project is released under the MIT License. Unless stated otherwise, files are retained from the original FOVI repository and covered by its MIT License (Copyright (c) 2026 Nicholas Blauch). See the `LICENSE` file for the full text and stacked copyright details.
