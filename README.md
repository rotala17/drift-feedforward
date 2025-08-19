# Representational Drift in Feedforward Network
This repository contains the code and data used to generate figures for:

**Nakamura, K., Endo, K., Kazama, H.** "Representational drift under synaptic and potential fluctuation affect differentially on discrimination learning" (2025).

## Repository Structure

```
├── note/                    # Jupyter notebooks for figure generation
├── data/                    # Precomputed data files
├── figure/                  # Generated figures output
├── src/                     # Simulation code
├── Project.toml            # Julia project dependencies
├── Manifest.toml           # Julia dependency manifest
└── README.md               # This file
```

## Setup Instructions

1. Clone this repository
2. Open any notebook in the `note/` directory
3. Run the first cell to install dependencies:
   ```julia
   using Pkg
   Pkg.activate("../Project.toml")
   Pkg.instantiate()
   ```
4. Run the remaining notebook cells in order to generate figures and data

## Generating Figures and Data

All figures can be reproduced by running the corresponding Jupyter notebooks in the `note/` directory:

| Figure | Notebook |
|--------|----------|
| Figure 1,S2 | `Fig1.ipynb` |
| Figures 2-4 | `Fig2-4.ipynb` |
| Figure 5,S4,S5 | `Fig5.ipynb` |
| Figure S1 | `FigS1.ipynb` |
| Figure S3 | `FigS3.ipynb` |

Most data is generated directly within the figure notebooks when you run them. Some computationally intensive cells include time information as comments based on the specified hardware environment.

**Exception: Figure 5 data** is precomputed and stored in the `data/` directory due to computational requirements (approximately 11 hours on the specified hardware). If you need to regenerate this data yourself, run `Simulate_Fig5.ipynb` (note: computation time may vary significantly depending on your hardware specifications).


## Requirements

- Julia
- Jupyter Notebook with IJulia kernel (For installation instructions, see: https://julialang.github.io/IJulia.jl/stable/manual/installation/)
- Dependencies specified in `Project.toml` and `Manifest.toml` (automatically installed via first notebook cell)

## Computational Environment

This analysis was performed using:
- **Julia version**: v1.11.6
- **Platform**: macOS Sonoma 14.1.2
- **Hardware**: MacBook Pro 2021 with Apple M1 Max chip, 64 GB memory

*Note: While the code should work on other platforms and Julia versions, results may vary slightly due to numerical precision differences.*
