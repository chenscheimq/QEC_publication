"""
QEC Configuration Module

Contains all shared constants, operators, and utility functions for
quantum error correction simulations with RAP protocols.
"""

import numpy as np
import matplotlib as mpl
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, basis, expect


class PlotConfig:
    """
    Publication-quality matplotlib configuration.

    Usage:
        PlotConfig.apply()  # Apply to current matplotlib session
    """

    # Font sizes
    FONT_SIZE = 12
    AXES_LABEL_SIZE = 14
    AXES_TITLE_SIZE = 16
    TICK_LABEL_SIZE = 12
    LEGEND_FONT_SIZE = 18

    # Line styles
    LINE_WIDTH = 2.5
    AXES_LINE_WIDTH = 1.2

    # Colors
    COLORS_LOGICAL = ['red', 'blue']
    COLORS_ERROR = ['gray']  # Error states in gray for clarity
    COLOR_ERROR_DEFAULT = 'black'

    # Gap annotation colors
    COLOR_GAP_RAP = 'purple'
    COLOR_GAP_QEC = 'darkgreen'

    @classmethod
    def apply(cls):
        """Apply publication-quality matplotlib settings."""
        mpl.rcParams['font.size'] = cls.FONT_SIZE
        mpl.rcParams['axes.labelsize'] = cls.AXES_LABEL_SIZE
        mpl.rcParams['axes.titlesize'] = cls.AXES_TITLE_SIZE
        mpl.rcParams['xtick.labelsize'] = cls.TICK_LABEL_SIZE
        mpl.rcParams['ytick.labelsize'] = cls.TICK_LABEL_SIZE
        mpl.rcParams['legend.fontsize'] = cls.LEGEND_FONT_SIZE
        mpl.rcParams['lines.linewidth'] = cls.LINE_WIDTH
        mpl.rcParams['axes.linewidth'] = cls.AXES_LINE_WIDTH


class QECConfig:
    """
    Configuration class for 3-qubit repetition code QEC simulations.

    This class provides:
    - Platform-specific parameters (IBM, QuEra)
    - Quantum operators for the 3-qubit repetition code
    - RAP pulse functions (Gaussian and sinusoidal)
    - Energy spectrum tracking utilities

    Parameters
    ----------
    platform : str, optional
        Platform to use: 'ibm' (default) or 'quera'
    omega_max : float, optional
        Maximum Rabi frequency in rad/s. If None, uses platform default.
    T_max : float, optional
        Total protocol time in seconds. If None, uses platform default.
    n_points : int, optional
        Number of time points for simulation. Default: 5001.

    Attributes
    ----------
    omega_max : float
        Maximum Rabi frequency [rad/s]
    T_max : float
        Total protocol time [s]
    t_list : ndarray
        Time grid for simulations
    I, X, Y, Z : Qobj
        Single-qubit Pauli operators
    X_L, Z_L, I_L : Qobj
        3-qubit logical operators
    S1, S2 : Qobj
        Stabilizer operators (Z_1 Z_2 and Z_2 Z_3)
    P_code : Qobj
        Code space projector
    logical_zero, logical_one : Qobj
        Logical basis states |0_L> = |000>, |1_L> = |111>

    Examples
    --------
    >>> config = QECConfig()  # IBM platform
    >>> config = QECConfig(platform='quera')  # QuEra platform
    >>> config = QECConfig(omega_max=2*np.pi*30e6, T_max=5e-6)  # Custom
    """

    # Platform presets
    PLATFORMS = {
        'ibm': {
            'omega_max': 2 * np.pi * 25e6,  # 25 MHz Rabi frequency
            'T_max': 4e-6,                   # 4 us total time
            'description': 'IBM superconducting qubits'
        },
        'quera': {
            'omega_max': 2 * np.pi * 4e6,   # 4 MHz Rabi frequency
            'T_max': 1e-6,                   # 1 us total time
            'description': 'QuEra neutral atoms'
        }
    }

    # Unit conversion factors
    TO_TIME_UNITS = 1e6           # s -> us
    TO_FREQ_UNITS = 1 / (2 * np.pi * 1e6)  # rad/s -> MHz
    TIME_UNIT_LABEL = r'$\mu$s'
    FREQ_UNIT_LABEL = 'MHz'

    def __init__(self, platform='ibm', omega_max=None, T_max=None, n_points=5001):
        """Initialize QEC configuration."""

        # Set platform parameters
        self.platform = platform.lower()
        if self.platform not in self.PLATFORMS:
            raise ValueError(f"Unknown platform: {platform}. "
                           f"Available: {list(self.PLATFORMS.keys())}")

        preset = self.PLATFORMS[self.platform]
        self.omega_max = omega_max if omega_max is not None else preset['omega_max']
        self.T_max = T_max if T_max is not None else preset['T_max']
        self.n_points = n_points

        # Time grid
        self.t_list = np.linspace(0, self.T_max, n_points)

        # Initialize operators
        self._init_operators()

    def _init_operators(self):
        """Initialize quantum operators for the 3-qubit repetition code."""

        # Single-qubit primitives
        self.I = qeye(2)
        self.X = sigmax()
        self.Y = sigmay()
        self.Z = sigmaz()

        # Logical operators (3-qubit repetition code)
        self.X_L = tensor(self.X, self.X, self.X)
        self.Z_L = tensor(self.Z, self.Z, self.Z)
        self.I_L = tensor(self.I, self.I, self.I)

        # Stabilizers for X-error protection
        self.S1 = tensor(self.Z, self.Z, self.I)  # Z_1 Z_2
        self.S2 = tensor(self.I, self.Z, self.Z)  # Z_2 Z_3

        # Logical basis states
        ket0 = basis(2, 0)
        ket1 = basis(2, 1)
        self.logical_zero = tensor(ket0, ket0, ket0)  # |000>
        self.logical_one = tensor(ket1, ket1, ket1)   # |111>

        # Code space projector: P = 1/4 * (I + S1)(I + S2)
        self.P_code = 0.25 * (self.I_L + self.S1) * (self.I_L + self.S2)

        # Hilbert space dimension
        self.dim = 2**3

    # =========================================================================
    # RAP Pulse Functions
    # =========================================================================

    def omega_gaussian(self, t):
        """
        Gaussian RAP pulse for Rabi frequency.

        Omega(t) = omega_max * exp(-0.5 * ((t - T/2) / sigma)^2)
        where sigma = T_max / 7

        Parameters
        ----------
        t : float or ndarray
            Time value(s) in seconds

        Returns
        -------
        float or ndarray
            Rabi frequency in rad/s
        """
        sigma = self.T_max / 7
        return self.omega_max * np.exp(-0.5 * ((t - self.T_max/2) / sigma)**2)

    def delta_linear(self, t):
        """
        Linear detuning sweep for RAP.

        Delta(t) = omega_max * (2t/T - 1)
        Sweeps from -omega_max to +omega_max

        Parameters
        ----------
        t : float or ndarray
            Time value(s) in seconds

        Returns
        -------
        float or ndarray
            Detuning in rad/s
        """
        return self.omega_max * (t / (self.T_max/2) - 1)

    def omega_sinusoidal(self, t):
        """
        Sinusoidal RAP pulse for transverse field.

        Omega(t) = omega_max * sin(pi * t / T)

        Parameters
        ----------
        t : float or ndarray
            Time value(s) in seconds

        Returns
        -------
        float or ndarray
            Transverse field amplitude in rad/s
        """
        return self.omega_max * np.sin(np.pi * t / self.T_max)

    def delta_sinusoidal(self, t):
        """
        Sinusoidal detuning sweep for RAP.

        Delta(t) = -omega_max * cos(pi * t / T)

        Parameters
        ----------
        t : float or ndarray
            Time value(s) in seconds

        Returns
        -------
        float or ndarray
            Longitudinal field in rad/s
        """
        return -self.omega_max * np.cos(np.pi * t / self.T_max)

    # =========================================================================
    # Hamiltonian Builders
    # =========================================================================

    def H_rap(self, t, Ep=0, pulse_type='gaussian'):
        """
        Build the RAP Hamiltonian at time t.

        H(t) = Omega(t) * X_L + Delta(t) * Z_L - Ep * (S1 + S2)

        Parameters
        ----------
        t : float
            Time in seconds
        Ep : float, optional
            Penalty energy in rad/s. Default: 0.
        pulse_type : str, optional
            Pulse shape: 'gaussian' or 'sinusoidal'. Default: 'gaussian'.

        Returns
        -------
        Qobj
            Hamiltonian at time t
        """
        if pulse_type == 'gaussian':
            omega = self.omega_gaussian(t)
            delta = self.delta_linear(t)
        elif pulse_type == 'sinusoidal':
            omega = self.omega_sinusoidal(t)
            delta = self.delta_sinusoidal(t)
        else:
            raise ValueError(f"Unknown pulse_type: {pulse_type}")

        H = self.X_L * omega + self.Z_L * delta - Ep * (self.S1 + self.S2)
        return H.to('csr')

    def make_H_func(self, Ep=0, pulse_type='gaussian'):
        """
        Create a Hamiltonian function H(t) for a given Ep.

        Parameters
        ----------
        Ep : float, optional
            Penalty energy in rad/s. Default: 0.
        pulse_type : str, optional
            Pulse shape: 'gaussian' or 'sinusoidal'. Default: 'gaussian'.

        Returns
        -------
        callable
            Function H(t) returning the Hamiltonian at time t
        """
        def H_func(t):
            return self.H_rap(t, Ep=Ep, pulse_type=pulse_type)
        return H_func

    # =========================================================================
    # Energy Spectrum Analysis
    # =========================================================================

    @staticmethod
    def _overlap_prob(v, ket):
        """Compute overlap probability |<ket|v>|^2."""
        return abs((ket.dag() * v))**2

    def track_code_eigenvalues(self, H_func, return_indices=False):
        """
        Track energy eigenvalues throughout the RAP protocol.

        Identifies the two code-space states by their overlap with the
        code projector, and assigns them to |0_L> or |1_L> based on
        instantaneous logical content. This allows tracking through
        avoided crossings where the character swaps.

        Parameters
        ----------
        H_func : callable
            Function H(t) returning Hamiltonian at time t
        return_indices : bool, optional
            If True, also return eigenstate indices. Default: False.

        Returns
        -------
        energies : dict
            Dictionary mapping index -> list of energies.
            0, 1 are code-space states; 2-7 are error states.
        idx_series : tuple of lists, optional
            If return_indices=True, returns (idx_0L, idx_1L) tracking
            which eigenstate index corresponds to each logical state.
        """
        energies = {i: [] for i in range(self.dim)}
        idx_series_0, idx_series_1 = [], []

        for ti, t in enumerate(self.t_list):
            evals, evecs = H_func(t).eigenstates()

            # Find code-space states by projector overlap
            pc = np.array([float(np.real(expect(self.P_code, v))) for v in evecs])
            code_candidates = list(np.argsort(-pc)[:2])
            a, b = sorted(code_candidates, key=lambda j: float(evals[j]))

            # Assign by instantaneous logical content
            w0a = self._overlap_prob(evecs[a], self.logical_zero)
            w1a = self._overlap_prob(evecs[a], self.logical_one)
            w0b = self._overlap_prob(evecs[b], self.logical_zero)
            w1b = self._overlap_prob(evecs[b], self.logical_one)

            if (w0a + w1b) >= (w0b + w1a):
                idx0L, idx1L = a, b
            else:
                idx0L, idx1L = b, a

            energies[0].append(float(evals[idx0L]))
            energies[1].append(float(evals[idx1L]))
            idx_series_0.append(idx0L)
            idx_series_1.append(idx1L)

            # Error states (remaining eigenstates, sorted by energy)
            rest = sorted([j for j in range(self.dim) if j not in (idx0L, idx1L)],
                         key=lambda j: float(evals[j]))
            for k, j in enumerate(rest, start=2):
                energies[k].append(float(evals[j]))

        # Convert to arrays
        for i in energies:
            energies[i] = np.array(energies[i])

        if return_indices:
            return energies, (idx_series_0, idx_series_1)
        return energies

    @staticmethod
    def break_at_swaps(y_vals, idx_series):
        """
        Insert NaN at points where eigenstate assignment changes.

        This prevents vertical line segments when plotting energy
        levels that swap character during the protocol.

        Parameters
        ----------
        y_vals : array-like
            Energy values to break
        idx_series : array-like
            Eigenstate indices at each time point

        Returns
        -------
        ndarray
            Energy values with NaN inserted at swap points
        """
        y_broken = np.array(y_vals, dtype=float).copy()
        idx_arr = np.array(idx_series, dtype=int)

        # Find where assignment changes
        swaps = np.where(idx_arr[1:] != idx_arr[:-1])[0] + 1

        # Insert NaN at swap points
        for s in swaps:
            if 0 < s < len(y_broken):
                y_broken[s] = np.nan

        return y_broken

    def compute_gaps(self, energies_no_penalty, energies_with_penalty):
        """
        Compute RAP gap and QEC gap from energy spectra.

        Parameters
        ----------
        energies_no_penalty : dict
            Energies from track_code_eigenvalues with Ep=0
        energies_with_penalty : dict
            Energies from track_code_eigenvalues with Ep>0

        Returns
        -------
        dict
            Dictionary with:
            - 'delta_rap': Minimum gap between logical states (Ep=0)
            - 't_delta_rap': Time at minimum RAP gap
            - 'delta_min_ep': Minimum code-to-error gap (Ep>0)
            - 't_delta_min_ep': Time at minimum QEC gap
        """
        # Convert to MHz for output
        E0_no = energies_no_penalty[0] * self.TO_FREQ_UNITS
        E1_no = energies_no_penalty[1] * self.TO_FREQ_UNITS

        gap_rap = np.abs(E1_no - E0_no)
        delta_rap = np.min(gap_rap)
        t_delta_rap = self.t_list[np.argmin(gap_rap)] * self.TO_TIME_UNITS

        # QEC gap: code-to-error separation
        E0_pen = energies_with_penalty[0] * self.TO_FREQ_UNITS
        E1_pen = energies_with_penalty[1] * self.TO_FREQ_UNITS
        max_code = np.maximum(E0_pen, E1_pen)

        error_energies = [energies_with_penalty[i] * self.TO_FREQ_UNITS
                         for i in range(2, self.dim)]
        min_error = np.min(error_energies, axis=0)

        gap_qec = min_error - max_code
        delta_min_ep = np.min(gap_qec)
        t_delta_min_ep = self.t_list[np.argmin(gap_qec)] * self.TO_TIME_UNITS

        return {
            'delta_rap': delta_rap,
            't_delta_rap': t_delta_rap,
            'delta_min_ep': delta_min_ep,
            't_delta_min_ep': t_delta_min_ep
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def to_MHz(self, value_rad_s):
        """Convert from rad/s to MHz."""
        return value_rad_s * self.TO_FREQ_UNITS

    def to_us(self, value_s):
        """Convert from seconds to microseconds."""
        return value_s * self.TO_TIME_UNITS

    def Ep_MHz_to_rad(self, Ep_MHz):
        """Convert penalty energy from MHz to rad/s."""
        return Ep_MHz * 2 * np.pi * 1e6

    def info(self):
        """Print configuration summary."""
        print("=" * 60)
        print(f"QEC Configuration ({self.platform.upper()} Platform)")
        print("=" * 60)
        print(f"  omega_max = {self.to_MHz(self.omega_max):.1f} MHz")
        print(f"  T_max     = {self.to_us(self.T_max):.1f} us")
        print(f"  n_points  = {self.n_points}")
        print("-" * 60)
        print("  Code: 3-qubit repetition [[3,1,3]]")
        print("  Logical states: |0_L> = |000>, |1_L> = |111>")
        print("  Stabilizers: S1 = Z1*Z2, S2 = Z2*Z3")
        print("=" * 60)
