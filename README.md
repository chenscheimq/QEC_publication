# QEC Publication: RAP Protocol for Quantum Error Correction

Publication-ready code for simulating Rapid Adiabatic Passage (RAP) protocols with quantum error correcting codes.

## Overview

This repository contains simulation tools and analysis notebooks for studying energy spectrum evolution under RAP protocols with penalty terms that confine dynamics to the logical code subspace.

**Supported QEC Codes:**
- **3-qubit Repetition Code [[3,1,3]]** — Stabilizer code with Z-type stabilizers
- **Bacon-Shor [[4,1,1,2]]** — Subsystem code with gauge penalization

## Repository Structure

```
QEC_publication/
├── qec_config/           # Core configuration module
│   ├── __init__.py
│   └── config.py         # QECConfig, BaconShorConfig, PlotConfig classes
├── notebooks/
│   ├── main_energy_spectrum_comparison.ipynb  # Main paper figure
│   ├── RAP_QEC_gaps_thesis.ipynb              # 3-qubit analysis
│   └── RAP_BaconShor_subsystem.ipynb          # Bacon-Shor analysis
├── data/                 # Cached simulation data (.npz files)
├── figs/                 # Generated figures
│   └── main_figure/      # Main paper figure (PDF, SVG, PNG)
└── README.md
```

## Installation

```bash
pip install numpy scipy matplotlib qutip
```

## Usage

### Quick Start

```python
from qec_config import QECConfig, BaconShorConfig, PlotConfig

# Apply publication-quality plot settings
PlotConfig.apply()

# Initialize 3-qubit repetition code
rep = QECConfig(n_points=1001)

# Initialize Bacon-Shor subsystem code
bs = BaconShorConfig(n_points=1001)

# Compute energy spectrum with penalty Ep=50 MHz
Ep = rep.Ep_MHz_to_rad(50)
energies, idx_series = rep.get_or_compute_spectrum(Ep=Ep)
```

### RAP Parameters

Default parameters used for simulations:

| Parameter | Value |
|-----------|-------|
| `omega_max` | 2π × 25 MHz (Rabi frequency) |
| `T_max` | 4 μs (protocol duration) |

### Data Caching

Computed spectra are automatically cached in the `data/` folder. Subsequent runs with identical parameters load from cache:

```python
# First call: computes and saves to data/
energies, idx = config.get_or_compute_spectrum(Ep=Ep)

# Second call: loads from cache
energies, idx = config.get_or_compute_spectrum(Ep=Ep)
```

## Module: `qec_config`

### Classes

| Class | Description |
|-------|-------------|
| `QECConfigBase` | Abstract base class with shared RAP functionality |
| `QECConfig` | 3-qubit repetition code [[3,1,3]] |
| `BaconShorConfig` | Bacon-Shor [[4,1,1,2]] subsystem code |
| `PlotConfig` | Matplotlib settings for Quantum journal style |

### Key Methods

```python
# RAP pulse shapes
config.omega_gaussian(t)  # Gaussian envelope
config.delta_linear(t)    # Linear detuning sweep

# Hamiltonian
H = config.H_rap(t, Ep=Ep, pulse_type='gaussian')

# Spectrum tracking through avoided crossings
energies, idx_series = config.track_code_eigenvalues(Ep=Ep)

# Break curves at swap points for clean plotting
E_broken = config.break_at_swaps(E, idx_series)
```

## Notebooks

### `main_energy_spectrum_comparison.ipynb`
Main figure for publication showing side-by-side energy spectra of both codes with state labels.

### `RAP_QEC_gaps_thesis.ipynb`
Detailed analysis of the 3-qubit repetition code including gap analysis.

### `RAP_BaconShor_subsystem.ipynb`
Analysis of the Bacon-Shor subsystem code with gauge penalization.

## Code Physics

### RAP Hamiltonian

The RAP Hamiltonian with penalty term:

```
H(t) = Ω(t) · X_L + Δ(t) · Z_L - E_p · H_penalty
```

where:
- `Z_L`, `X_L` are logical Pauli operators
- `H_penalty` penalizes states outside the code space
- For repetition code: `H_penalty = S1 + S2` (stabilizers)
- For Bacon-Shor: `H_penalty = Σ_i G_i` (gauge operators)

### Operators

**Repetition Code [[3,1,3]]:**
- Logical: `X_L = XXX`, `Z_L = ZZZ`
- Stabilizers: `S1 = ZZI`, `S2 = IZZ`

**Bacon-Shor [[4,1,1,2]]:**
- Logical: `X_L = XIXI`, `Z_L = ZZII`
- Gauge: `G = {XXII, ZIZI, IIXX, IZIZ}`

## Output Formats

Figures are saved in multiple formats:
- **PDF** — Vector format for publication
- **SVG** — Vector format for editing
- **PNG** — Raster format (300 DPI)
- **NPZ** — Raw data for reproducibility

## License

MIT License

## Citation

If you use this code, please cite the associated publication.
