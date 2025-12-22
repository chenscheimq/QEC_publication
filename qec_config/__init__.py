"""
QEC Publication Configuration Module

This module provides shared configuration, constants, operators, and utility functions
for quantum error correction simulations with RAP (Rapid Adiabatic Passage) protocols.

Supported codes:
- 3-qubit repetition code [[3,1,3]] (QECConfig)
- Bacon-Shor [[4,1,1,2]] subsystem code (BaconShorConfig)

Usage:
    from qec_config import QECConfig, BaconShorConfig

    # 3-qubit repetition code
    config = QECConfig()  # IBM platform
    config = QECConfig(platform='quera')  # QuEra platform

    # Bacon-Shor subsystem code
    bs_config = BaconShorConfig()
"""

from .config import QECConfig, BaconShorConfig, PlotConfig

__all__ = ['QECConfig', 'BaconShorConfig', 'PlotConfig']
__version__ = '1.1.0'
