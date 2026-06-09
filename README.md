# Square-FOVI

Square-FOVI is an extended foveated interface for deep vision models. Building upon the foundational FOVI architecture, this project introduces a novel geometry and patching mechanism based on concentric square sampling and Chebyshev topologies.

## Core Architectural Extensions

This repository implements the following primary architectural modules:

* **Concentric Square Node Sampling:** Replaces radial sampling with an equal-area square geometry, snapping nodes to corners and edges.
* **DoB Receptive Fields & SAT Hypernetwork:** Replaces the standard receptive field with a hypernetwork generating $k \times 3 \times 2$ parameters (using SiLU and tanh) for a Difference of Gaussians/Boxes (DoB) filter.
* **Chebyshev Manifold Patching:** Utilizes Chebyshev ($L_\infty$) kNN on the manifold for native kernel correspondence and bilinear interpolation for idiosyncratic spatial positions.
* **Log-Chebyshev P-RoPE:** Implements custom positional embeddings replacing axial RoPE for DINOv3 backbones.
* **Semantic Fixation Points:** Uses heavily downsampled early DINOv3 layers to extract semantic interest points for dynamic fixation.

## Installation

```bash
git clone [your-repo-url]
cd [your-repo-name]
pip install -e .
```

## Credits and Acknowledgements

This project, **Square-FOVI**, is a standalone hard fork built upon the foundational work of the original [FOVI repository](https://github.com/nblauch/fovi). 

* The core architecture and initial implementation were developed by **Nicholas Blauch**.
* Architectural modifications—including the concentric square sampling, Chebyshev manifold patching, custom receptive fields, P-RoPE positional embeddings, semantic fixation policies, and custom profiling tools—were developed by **Arthur Solère (2026)**.

## License

This project is released under the MIT License.

**License Note:** Unless explicitly stated otherwise at the top of a file, all files in this repository are retained as-is from the original FOVI repository and are covered under the original MIT License (Copyright (c) 2026 Nicholas Blauch). Files modified or created for the Square-FOVI architecture are explicitly marked in their respective file headers. See the `LICENSE` file for the full text and stacked copyright details.