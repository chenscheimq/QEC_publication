"""
QEC Configuration Module

Contains all shared constants, operators, and utility functions for
quantum error correction simulations with RAP protocols.
"""

import numpy as np
import matplotlib as mpl
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, basis, expect
from pathlib import Path
import hashlib
import json

from .utils import break_at_swaps as _break_at_swaps


class PlotConfig:
    """
    Publication-quality matplotlib configuration for Quantum journal.

    Follows best practices for physics publications:
    - Clean serif fonts compatible with LaTeX documents
    - Consistent sizing for single/double column figures
    - High-quality vector output (PDF)

    Usage:
        PlotConfig.apply()  # Apply to current matplotlib session
    """

    # Quantum journal column widths (approximate, in inches)
    SINGLE_COLUMN_WIDTH = 3.4  # ~86mm
    DOUBLE_COLUMN_WIDTH = 7.0  # ~178mm

    # Font sizes (optimized for readability at publication size)
    FONT_SIZE = 14
    AXES_LABEL_SIZE = 16
    AXES_TITLE_SIZE = 16
    TICK_LABEL_SIZE = 14
    LEGEND_FONT_SIZE = 12

    # Line styles
    LINE_WIDTH = 1.5
    AXES_LINE_WIDTH = 0.8
    GRID_LINE_WIDTH = 0.5

    # Marker sizes
    MARKER_SIZE = 4

    # Colors - colorblind-friendly palette
    COLORS_LOGICAL = ['#E41A1C', '#377EB8']  # Red, Blue (ColorBrewer)
    COLORS_ERROR = ['#999999']  # Gray for error states
    COLOR_ERROR_DEFAULT = '#666666'

    # Gap annotation colors
    COLOR_GAP_RAP = '#984EA3'  # Purple
    COLOR_GAP_QEC = '#4DAF4A'  # Green

    # Additional colors for extended palettes
    COLORS_EXTENDED = [
        '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
        '#FF7F00', '#FFFF33', '#A65628', '#F781BF'
    ]

    @classmethod
    def apply(cls, use_latex=False):
        """
        Apply publication-quality matplotlib settings.

        Parameters
        ----------
        use_latex : bool, optional
            If True, use LaTeX for text rendering (requires LaTeX installation).
            Default: False (uses mathtext which is more portable)
        """
        # Font configuration
        mpl.rcParams['font.size'] = cls.FONT_SIZE
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Computer Modern Roman', 'DejaVu Serif',
                                       'Times New Roman', 'serif']

        # Use LaTeX if requested and available
        if use_latex:
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        else:
            mpl.rcParams['text.usetex'] = False
            mpl.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern

        # Axes configuration
        mpl.rcParams['axes.labelsize'] = cls.AXES_LABEL_SIZE
        mpl.rcParams['axes.titlesize'] = cls.AXES_TITLE_SIZE
        mpl.rcParams['axes.linewidth'] = cls.AXES_LINE_WIDTH
        mpl.rcParams['axes.labelpad'] = 4

        # Tick configuration
        mpl.rcParams['xtick.labelsize'] = cls.TICK_LABEL_SIZE
        mpl.rcParams['ytick.labelsize'] = cls.TICK_LABEL_SIZE
        mpl.rcParams['xtick.major.width'] = cls.AXES_LINE_WIDTH
        mpl.rcParams['ytick.major.width'] = cls.AXES_LINE_WIDTH
        mpl.rcParams['xtick.minor.width'] = cls.AXES_LINE_WIDTH * 0.6
        mpl.rcParams['ytick.minor.width'] = cls.AXES_LINE_WIDTH * 0.6
        mpl.rcParams['xtick.major.size'] = 4
        mpl.rcParams['ytick.major.size'] = 4
        mpl.rcParams['xtick.minor.size'] = 2
        mpl.rcParams['ytick.minor.size'] = 2
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'

        # Legend configuration
        mpl.rcParams['legend.fontsize'] = cls.LEGEND_FONT_SIZE
        mpl.rcParams['legend.framealpha'] = 0.9
        mpl.rcParams['legend.edgecolor'] = '0.8'
        mpl.rcParams['legend.borderpad'] = 0.4
        mpl.rcParams['legend.handlelength'] = 1.5

        # Line configuration
        mpl.rcParams['lines.linewidth'] = cls.LINE_WIDTH
        mpl.rcParams['lines.markersize'] = cls.MARKER_SIZE

        # Grid configuration
        mpl.rcParams['grid.linewidth'] = cls.GRID_LINE_WIDTH
        mpl.rcParams['grid.alpha'] = 0.3

        # Figure configuration
        mpl.rcParams['figure.dpi'] = 150
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['savefig.format'] = 'pdf'
        mpl.rcParams['savefig.bbox'] = 'tight'
        mpl.rcParams['savefig.pad_inches'] = 0.05

        # PDF backend settings for vector output
        mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts in PDF

    @classmethod
    def figure_single_column(cls, aspect_ratio=0.75):
        """Create a figure sized for single column width."""
        import matplotlib.pyplot as plt
        width = cls.SINGLE_COLUMN_WIDTH
        height = width * aspect_ratio
        return plt.figure(figsize=(width, height))

    @classmethod
    def figure_double_column(cls, aspect_ratio=0.4):
        """Create a figure sized for double column width."""
        import matplotlib.pyplot as plt
        width = cls.DOUBLE_COLUMN_WIDTH
        height = width * aspect_ratio
        return plt.figure(figsize=(width, height))


class SimulationParams:
    """
    Simulation parameters for thermal noise fidelity calculations.

    These are IBM-platform parameters based on dilution fridge conditions.

    Usage:
        from qec_config import SimulationParams
        params = SimulationParams()
        lambda_2 = params.LAMBDA_2
        temperatures = params.TEMPERATURES
    """

    # Thermal noise coupling strength (IBM platform)
    LAMBDA_2 = 1e4

    # Physical temperature (Kelvin) - default for single-T simulations
    T_PHYS_K = 0.015

    # Temperature sweep for multi-T simulations (Kelvin)
    # Dilution fridge regime: 8-50 mK
    TEMPERATURES = np.array([0.008, 0.010, 0.015, 0.020, 0.030, 0.050], dtype=float)

    # Penalty energy sweep (MHz) - converted to rad/s when used
    EP_MIN_MHZ = 10
    EP_MAX_MHZ = 250
    N_EP_POINTS = 10

    # Time discretization for simulations
    N_POINTS = 101

    @classmethod
    def get_Ep_sweep_rad(cls):
        """Return Ep sweep values in rad/s."""
        return 2 * np.pi * np.linspace(cls.EP_MIN_MHZ, cls.EP_MAX_MHZ, cls.N_EP_POINTS) * 1e6

    @classmethod
    def get_Ep_sweep_MHz(cls):
        """Return Ep sweep values in MHz."""
        return np.linspace(cls.EP_MIN_MHZ, cls.EP_MAX_MHZ, cls.N_EP_POINTS)

    @classmethod
    def info(cls):
        """Print simulation parameters."""
        print("=" * 60)
        print("Simulation Parameters (IBM Platform)")
        print("=" * 60)
        print(f"  lambda_2     = {cls.LAMBDA_2:.0e}")
        print(f"  T_phys       = {cls.T_PHYS_K * 1e3:.1f} mK")
        print(f"  Ep range     = {cls.EP_MIN_MHZ} - {cls.EP_MAX_MHZ} MHz ({cls.N_EP_POINTS} points)")
        print(f"  Temperatures = {cls.TEMPERATURES * 1e3} mK")
        print(f"  n_points     = {cls.N_POINTS}")
        print("=" * 60)


class QECConfigBase:
    """
    Base configuration class for QEC simulations with RAP protocols.

    Provides shared functionality for all QEC codes:
    - RAP pulse functions (Gaussian and sinusoidal)
    - Energy spectrum tracking utilities
    - Data caching

    Subclasses must implement:
    - _init_operators(): Initialize code-specific operators
    - _get_penalty_term(): Return the penalty Hamiltonian
    - info(): Print code-specific information
    - code_name: Class attribute for cache file naming
    """

    # Default RAP parameters
    DEFAULT_OMEGA_MAX = 2 * np.pi * 25e6  # 25 MHz Rabi frequency
    DEFAULT_T_MAX = 4e-6                   # 4 us total time

    # Unit conversion factors
    TO_TIME_UNITS = 1e6           # s -> us
    TO_FREQ_UNITS = 1 / (2 * np.pi * 1e6)  # rad/s -> MHz
    TIME_UNIT_LABEL = r'$\mu$s'
    FREQ_UNIT_LABEL = 'MHz'

    # Subclasses should override
    code_name = 'base'

    def __init__(self, omega_max=None, T_max=None, n_points=1001):
        """Initialize QEC configuration."""
        self.omega_max = omega_max if omega_max is not None else self.DEFAULT_OMEGA_MAX
        self.T_max = T_max if T_max is not None else self.DEFAULT_T_MAX
        self.n_points = n_points
        self.t_list = np.linspace(0, self.T_max, n_points)

        # Single-qubit primitives (shared)
        self.I = qeye(2)
        self.X = sigmax()
        self.Y = sigmay()
        self.Z = sigmaz()

        # Code-specific initialization
        self._init_operators()

    def _init_operators(self):
        """Initialize code-specific operators. Override in subclass."""
        raise NotImplementedError

    def _get_penalty_term(self):
        """Return the penalty Hamiltonian term. Override in subclass."""
        raise NotImplementedError

    # =========================================================================
    # RAP Pulse Functions
    # =========================================================================

    def omega_gaussian(self, t):
        """Gaussian RAP pulse: Omega(t) = omega_max * exp(-0.5*((t-T/2)/sigma)^2)"""
        sigma = self.T_max / 7
        return self.omega_max * np.exp(-0.5 * ((t - self.T_max/2) / sigma)**2)

    def delta_linear(self, t):
        """Linear detuning sweep from -omega_max to +omega_max."""
        return self.omega_max * (t / (self.T_max/2) - 1)

    def omega_sinusoidal(self, t):
        """Sinusoidal RAP pulse: Omega(t) = omega_max * sin(pi*t/T)"""
        return self.omega_max * np.sin(np.pi * t / self.T_max)

    def delta_sinusoidal(self, t):
        """Sinusoidal detuning: Delta(t) = -omega_max * cos(pi*t/T)"""
        return -self.omega_max * np.cos(np.pi * t / self.T_max)

    # =========================================================================
    # Hamiltonian Builders
    # =========================================================================

    def H_rap(self, t, Ep=0, pulse_type='gaussian'):
        """Build RAP Hamiltonian: H(t) = Omega(t)*X_L + Delta(t)*Z_L - Ep*H_penalty"""
        if pulse_type == 'gaussian':
            omega = self.omega_gaussian(t)
            delta = self.delta_linear(t)
        elif pulse_type == 'sinusoidal':
            omega = self.omega_sinusoidal(t)
            delta = self.delta_sinusoidal(t)
        else:
            raise ValueError(f"Unknown pulse_type: {pulse_type}")

        H = self.X_L * omega + self.Z_L * delta - Ep * self._get_penalty_term()
        return H.to('csr')

    def make_H_func(self, Ep=0, pulse_type='gaussian'):
        """Create a Hamiltonian function H(t) for a given Ep."""
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
        """Track energy eigenvalues throughout the RAP protocol."""
        energies = {i: [] for i in range(self.dim)}
        idx_series_0, idx_series_1 = [], []

        for t in self.t_list:
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

            # Error states
            rest = sorted([j for j in range(self.dim) if j not in (idx0L, idx1L)],
                         key=lambda j: float(evals[j]))
            for k, j in enumerate(rest, start=2):
                energies[k].append(float(evals[j]))

        for i in energies:
            energies[i] = np.array(energies[i])

        if return_indices:
            return energies, (idx_series_0, idx_series_1)
        return energies

    @staticmethod
    def break_at_swaps(y_vals, idx_series):
        """Insert NaN at eigenstate assignment changes to break plot lines."""
        return _break_at_swaps(y_vals, idx_series)

    def compute_gaps(self, energies_no_penalty, energies_with_penalty):
        """Compute RAP gap and QEC gap from energy spectra."""
        E0_no = energies_no_penalty[0] * self.TO_FREQ_UNITS
        E1_no = energies_no_penalty[1] * self.TO_FREQ_UNITS

        gap_rap = np.abs(E1_no - E0_no)
        delta_rap = np.min(gap_rap)
        t_delta_rap = self.t_list[np.argmin(gap_rap)] * self.TO_TIME_UNITS

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
            'delta_rap': delta_rap, 't_delta_rap': t_delta_rap,
            'delta_min_ep': delta_min_ep, 't_delta_min_ep': t_delta_min_ep
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

    # =========================================================================
    # Data Caching
    # =========================================================================

    def _get_cache_key(self, Ep, pulse_type):
        """Generate a unique cache key for simulation parameters."""
        params = {
            'code': self.code_name,
            'omega_max': float(self.omega_max),
            'T_max': float(self.T_max),
            'n_points': self.n_points,
            'Ep': float(Ep),
            'pulse_type': pulse_type
        }
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:12]

    def get_data_path(self, Ep, pulse_type, data_dir=None):
        """Get the path for cached data file."""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / 'data'
        else:
            data_dir = Path(data_dir)

        Ep_MHz = self.to_MHz(Ep)
        cache_key = self._get_cache_key(Ep, pulse_type)
        filename = f"spectrum_{self.code_name}_{pulse_type}_Ep{Ep_MHz:.1f}MHz_{cache_key}.npz"
        return data_dir / filename

    def save_spectrum_data(self, energies, idx_series, Ep, pulse_type, data_dir=None):
        """Save computed energy spectrum data to disk."""
        filepath = self.get_data_path(Ep, pulse_type, data_dir)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            't_list': self.t_list,
            'idx_0L': np.array(idx_series[0]),
            'idx_1L': np.array(idx_series[1]),
        }
        for i, arr in energies.items():
            data[f'E{i}'] = arr

        metadata = {
            'code': self.code_name,
            'omega_max_MHz': self.to_MHz(self.omega_max),
            'T_max_us': self.to_us(self.T_max),
            'n_points': self.n_points,
            'Ep_MHz': self.to_MHz(Ep),
            'pulse_type': pulse_type,
            'dim': self.dim
        }
        data['metadata'] = np.array([json.dumps(metadata)])

        np.savez_compressed(filepath, **data)
        return filepath

    def load_spectrum_data(self, Ep, pulse_type, data_dir=None):
        """Load cached energy spectrum data from disk."""
        filepath = self.get_data_path(Ep, pulse_type, data_dir)

        if not filepath.exists():
            return None, None

        data = np.load(filepath)
        energies = {}
        for i in range(self.dim):
            key = f'E{i}'
            if key in data:
                energies[i] = data[key]

        idx_series = (list(data['idx_0L']), list(data['idx_1L']))
        return energies, idx_series

    def get_or_compute_spectrum(self, Ep, pulse_type='gaussian', data_dir=None,
                                 force_recompute=False, verbose=True):
        """Get energy spectrum from cache or compute if not available."""
        Ep_MHz = self.to_MHz(Ep)

        if not force_recompute:
            energies, idx_series = self.load_spectrum_data(Ep, pulse_type, data_dir)
            if energies is not None:
                if verbose:
                    print(f"  Loaded cached data for Ep = {Ep_MHz:.1f} MHz")
                return energies, idx_series

        if verbose:
            print(f"  Computing spectrum for Ep = {Ep_MHz:.1f} MHz...")

        H_func = self.make_H_func(Ep=Ep, pulse_type=pulse_type)
        energies, idx_series = self.track_code_eigenvalues(H_func, return_indices=True)

        filepath = self.save_spectrum_data(energies, idx_series, Ep, pulse_type, data_dir)
        if verbose:
            print(f"    Saved to: {filepath.name}")

        return energies, idx_series


class QECConfig(QECConfigBase):
    """
    Configuration for 3-qubit repetition code [[3,1,3]] QEC simulations.

    Examples
    --------
    >>> config = QECConfig()
    >>> config = QECConfig(n_points=2001)  # Higher resolution
    """

    code_name = 'rep3'

    def __init__(self, omega_max=None, T_max=None, n_points=1001):
        super().__init__(omega_max, T_max, n_points)

    def _init_operators(self):
        """Initialize operators for the 3-qubit repetition code."""
        ket0, ket1 = basis(2, 0), basis(2, 1)

        # Logical operators
        self.X_L = tensor(self.X, self.X, self.X)
        self.Z_L = tensor(self.Z, self.Z, self.Z)
        self.I_L = tensor(self.I, self.I, self.I)

        # Stabilizers
        self.S1 = tensor(self.Z, self.Z, self.I)  # Z_1 Z_2
        self.S2 = tensor(self.I, self.Z, self.Z)  # Z_2 Z_3

        # Logical basis states
        self.logical_zero = tensor(ket0, ket0, ket0)  # |000>
        self.logical_one = tensor(ket1, ket1, ket1)   # |111>

        # Code space projector
        self.P_code = 0.25 * (self.I_L + self.S1) * (self.I_L + self.S2)

        self.dim = 2**3

    def _get_penalty_term(self):
        """Return stabilizer penalty: S1 + S2"""
        return self.S1 + self.S2

    def info(self):
        """Print configuration summary."""
        print("=" * 60)
        print("3-Qubit Repetition Code [[3,1,3]]")
        print("=" * 60)
        print(f"  omega_max = {self.to_MHz(self.omega_max):.1f} MHz")
        print(f"  T_max     = {self.to_us(self.T_max):.1f} us")
        print(f"  n_points  = {self.n_points}")
        print("-" * 60)
        print("  Logical: |0_L> = |000>, |1_L> = |111>")
        print("  Stabilizers: S1 = Z1Z2, S2 = Z2Z3")
        print("=" * 60)


class BaconShorConfig(QECConfigBase):
    """
    Configuration for Bacon-Shor [[4,1,1,2]] subsystem code simulations.

    Qubit layout (2x2 grid):
        1 -- 2
        |    |
        3 -- 4

    Examples
    --------
    >>> config = BaconShorConfig()
    >>> config = BaconShorConfig(n_points=2001)  # Higher resolution
    """

    code_name = 'bacon_shor'

    def __init__(self, omega_max=None, T_max=None, n_points=1001):
        super().__init__(omega_max, T_max, n_points)

    def _init_operators(self):
        """Initialize operators for the Bacon-Shor [[4,1,1,2]] code."""
        ket0, ket1 = basis(2, 0), basis(2, 1)

        # 4-qubit identity
        self.I_L = tensor(self.I, self.I, self.I, self.I)

        # Logical operators
        self.X_L = tensor(self.X, self.I, self.X, self.I)  # X1 X3
        self.Z_L = tensor(self.Z, self.Z, self.I, self.I)  # Z1 Z2

        # Gauge (ancilla) operators
        self.X_A = tensor(self.I, self.X, self.I, self.X)  # X2 X4
        self.Z_A = tensor(self.I, self.I, self.Z, self.Z)  # Z3 Z4

        # Gauge generators
        self.G = [
            tensor(self.X, self.X, self.I, self.I),  # X1 X2
            tensor(self.Z, self.I, self.Z, self.I),  # Z1 Z3
            tensor(self.I, self.I, self.X, self.X),  # X3 X4
            tensor(self.I, self.Z, self.I, self.Z),  # Z2 Z4
        ]

        # Logical basis states
        self.logical_zero = (tensor(ket0, ket0, ket0, ket0) +
                            tensor(ket1, ket1, ket1, ket1)).unit()
        self.logical_one = (self.X_L * self.logical_zero).unit()

        # Gauge qubit states
        self.ancilla_zero = (tensor(ket0, ket0, ket1, ket1) +
                            tensor(ket1, ket1, ket0, ket0)).unit()
        self.ancilla_one = (self.X_A * self.ancilla_zero).unit()

        # Code space projector
        self.P_code = self.I_L
        for g in self.G:
            self.P_code = self.P_code * (self.I_L + g) / 2

        self.dim = 2**4

    def _get_penalty_term(self):
        """Return gauge penalty: sum(G)"""
        return sum(self.G)

    def track_logical_projected(self, Ep, pulse_type='gaussian'):
        """
        Track logical states by projecting H onto the 2D logical subspace.

        Useful for Ep=0 where full tracking can be noisy.
        """
        H_func = self.make_H_func(Ep=Ep, pulse_type=pulse_type)
        basis_states = [self.logical_zero, self.logical_one]

        E0_list, E1_list = [], []
        idx_series_0, idx_series_1 = [], []

        for t in self.t_list:
            H_t = H_func(t)

            # Build 2x2 matrix in logical basis
            H_sub = np.zeros((2, 2), dtype=complex)
            for i, ket_i in enumerate(basis_states):
                for j, ket_j in enumerate(basis_states):
                    element = ket_i.dag() * H_t * ket_j
                    if hasattr(element, 'full'):
                        H_sub[i, j] = element.full()[0, 0]
                    else:
                        H_sub[i, j] = complex(element)

            evals, vecs = np.linalg.eigh(H_sub)

            # Assign by logical overlap
            w0a, w1a = abs(vecs[0, 0])**2, abs(vecs[1, 0])**2
            w0b, w1b = abs(vecs[0, 1])**2, abs(vecs[1, 1])**2

            if (w0a + w1b) >= (w0b + w1a):
                idx0, idx1 = 0, 1
            else:
                idx0, idx1 = 1, 0

            E0_list.append(float(evals[idx0]))
            E1_list.append(float(evals[idx1]))
            idx_series_0.append(idx0)
            idx_series_1.append(idx1)

        return np.array(E0_list), np.array(E1_list), (idx_series_0, idx_series_1)

    def info(self):
        """Print configuration summary."""
        print("=" * 60)
        print("Bacon-Shor [[4,1,1,2]] Subsystem Code")
        print("=" * 60)
        print(f"  omega_max = {self.to_MHz(self.omega_max):.1f} MHz")
        print(f"  T_max     = {self.to_us(self.T_max):.1f} us")
        print(f"  n_points  = {self.n_points}")
        print("-" * 60)
        print("  Logical: |0_L> = (|0000>+|1111>)/sqrt(2)")
        print("  Gauge: X1X2, Z1Z3, X3X4, Z2Z4")
        print("=" * 60)
