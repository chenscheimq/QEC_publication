"""
QEC Publication Configuration Module

This module provides shared configuration, constants, operators, and utility functions
for quantum error correction simulations, specifically for the 3-qubit repetition code
with RAP (Rapid Adiabatic Passage) protocols.

Usage:
    from qec_config import QECConfig

    config = QECConfig()  # Uses default IBM platform parameters
    # or
    config = QECConfig(platform='quera')  # Use QuEra platform parameters
"""

from .config import QECConfig, PlotConfig

__all__ = ['QECConfig', 'PlotConfig']
__version__ = '1.0.0'
