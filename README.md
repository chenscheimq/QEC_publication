# QEC Publication: RAP Protocol for Quantum Error Correction

Publication-ready code for simulating Rapid Adiabatic Passage (RAP) protocols with quantum error correcting codes.

## Overview

This repository contains simulation tools and analysis notebooks for studying energy spectrum evolution under RAP protocols with penalty terms that confine dynamics to the logical code subspace.

**Supported QEC Codes:**
- **3-qubit Repetition Code [[3,1,3]]** — Stabilizer code with Z-type stabilizers
- **Bacon-Shor [[4,1,1,2]]** — Subsystem code with gauge penalization

**Noise Models:**
- **Thermal Noise** — Open-system dynamics with Lindblad master equation
- **Control Noise (1/f-like)** — Ornstein-Uhlenbeck noise on drive fields

## Repository Structure

```
QEC_publication/
├── qec_config/              # Core configuration module
│   ├── __init__.py
│   ├── config.py            # QECConfig, BaconShorConfig, PlotConfig, SimulationParams
│   └── utils.py             # Thermal noise, OU noise, simulation helpers
├── notebooks/
│   ├── main_energy_spectrum_comparison.ipynb   # Energy spectrum figure
│   ├── main_fidelity_vs_Ep.ipynb               # Thermal noise fidelity
│   ├── main_local_controller_noise.ipynb       # 1/f control noise analysis
│   └── reference/           # Reference/development notebooks
├── data/                    # Cached simulation data (.npz files)
│   ├── control_noise/       # Control noise simulation data
│   └── thermal_noise/       # Thermal noise simulation data
├── figs/                    # Generated figures
│   ├── main_figure/         # Energy spectrum figures
│   ├── thermal_noise/       # Thermal noise figures
│   └── control_noise/       # Control noise figures
└── README.md
```

## Installation

```bash
pip install numpy scipy matplotlib qutip tqdm
```

## Usage

### Quick Start

```python
from qec_config import QECConfig, BaconShorConfig, PlotConfig, SimulationParams

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

### Thermal Noise Simulation

```python
from qec_config import simulate_rap_fidelity, SimulationParams

# Simulate RAP with thermal bath coupling
result = simulate_rap_fidelity(
    rep, rep.logical_zero, rep.logical_one,
    Ep=rep.Ep_MHz_to_rad(75),
    T_K=0.015,  # 15 mK
    lambda_2=SimulationParams.LAMBDA_2
)
print(f"Fidelity: {result['fidelity']:.4f}")
```

### Control Noise (1/f-like) Simulation

```python
from qec_config import simulate_control_noise_fidelity

# Simulate RAP with OU control noise
result = simulate_control_noise_fidelity(
    rep, Ep=rep.Ep_MHz_to_rad(100),
    rms_X_frac=0.20,  # 20% RMS noise on drive
    tau_X=0.5 * rep.T_max,  # Correlation time
    n_realizations=10
)
print(f"Mean fidelity: {result['fidelity_mean']:.4f} ± {result['fidelity_std']:.4f}")
```

### RAP Parameters

Default parameters (IBM platform):

| Parameter | Value |
|-----------|-------|
| `omega_max` | 2π × 25 MHz (Rabi frequency) |
| `T_max` | 4 μs (protocol duration) |
| `lambda_2` | 1e4 (thermal coupling) |
| `T_phys` | 15 mK (fridge temperature) |

### Data Caching

Computed spectra are automatically cached in the `data/` folder:

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
| `QECConfig` | 3-qubit repetition code [[3,1,3]] |
| `BaconShorConfig` | Bacon-Shor [[4,1,1,2]] subsystem code |
| `PlotConfig` | Matplotlib settings for publication quality |
| `SimulationParams` | IBM platform simulation parameters |

### Key Functions

```python
# Thermal noise simulation
simulate_rap_fidelity(config, initial, target, Ep, T_K, lambda_2)
simulate_fidelity_sweep(config, Ep_values, T_values, lambda_2)

# Control noise (1/f-like) simulation
sigma_from_rms(target_rms, tau)  # Convert RMS to OU sigma
ou_trace(times, tau, sigma)  # Generate OU noise trace
simulate_control_noise_fidelity(config, Ep, rms_X_frac, tau_X)
compute_Ep_vs_rms_curve(config, tau_X, rms_fracs, Ep_grid, target_fid)

# Thermal bath operators
c_ops_gen_thermal(t_eval, H_func, n_qubits, lambda_2, T_K)
```

## Notebooks

### Main Publication Figures

| Notebook | Description |
|----------|-------------|
| `main_energy_spectrum_comparison.ipynb` | Energy spectra of both codes with state labels |
| `main_fidelity_vs_Ep.ipynb` | Fidelity vs penalty energy under thermal noise |
| `main_local_controller_noise.ipynb` | Required Ep vs RMS noise amplitude |

### Reference Notebooks

Development and reference notebooks are in `notebooks/reference/`.

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

### Noise Models

**Thermal Noise:** Lindblad master equation with thermal collapse operators based on instantaneous Hamiltonian eigenstates.

**Control Noise:** Ornstein-Uhlenbeck process modeling 1/f-like fluctuations in drive amplitude and detuning:
```
dx = -(x/τ)dt + σ√dt · ξ
```
where τ is correlation time and σ is noise strength (related to RMS by σ = √2·RMS/√τ).

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
