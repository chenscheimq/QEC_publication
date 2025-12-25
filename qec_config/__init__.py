"""
QEC Publication Configuration Module

This module provides shared configuration, constants, operators, and utility functions
for quantum error correction simulations with RAP (Rapid Adiabatic Passage) protocols.

Supported codes:
- 3-qubit repetition code [[3,1,3]] (QECConfig)
- Bacon-Shor [[4,1,1,2]] subsystem code (BaconShorConfig)

Usage:
    from qec_config import QECConfig, BaconShorConfig, PlotConfig

    # 3-qubit repetition code
    config = QECConfig()
    config = QECConfig(n_points=2001)  # Higher resolution

    # Bacon-Shor subsystem code
    bs_config = BaconShorConfig()

    # Thermal noise model for open-system simulations
    from qec_config.utils import c_ops_gen_thermal, bose_einstein_N
"""

from .config import QECConfig, BaconShorConfig, PlotConfig, SimulationParams
from .utils import (
    c_ops_gen_thermal,
    c_ops_gen_thermal_full,
    bose_einstein_N,
    spectral_amp,
    MHz_to_rad_s,
    rad_s_to_MHz,
    us_to_s,
    s_to_us,
    break_at_swaps,
    simulate_rap_fidelity,
    simulate_fidelity_sweep,
)

__all__ = [
    'QECConfig',
    'BaconShorConfig',
    'PlotConfig',
    'SimulationParams',
    # Noise model
    'c_ops_gen_thermal',
    'c_ops_gen_thermal_full',
    'bose_einstein_N',
    'spectral_amp',
    # Unit conversions
    'MHz_to_rad_s',
    'rad_s_to_MHz',
    'us_to_s',
    's_to_us',
    # Plotting utilities
    'break_at_swaps',
    # Simulation helpers
    'simulate_rap_fidelity',
    'simulate_fidelity_sweep',
]
__version__ = '1.2.0'
