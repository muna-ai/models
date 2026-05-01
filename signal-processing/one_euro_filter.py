#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna"]
# ///

from muna import compile, Parameter
from numpy import abs, ndarray, pi, zeros
from typing import Annotated

@compile()
def one_euro_filter(
    data: Annotated[
        ndarray,
        Parameter.Generic(description="Input data to smooth with shape (N,).")
    ],
    state: Annotated[
        ndarray,
        Parameter.Generic(description="Internal filter state with shape (2*N + 1,). ")
    ],
    delta_time: Annotated[float, Parameter.Numeric(
        description="Seconds elapsed since the previous sample.",
        min=1e-6,
    )],
    cutoff_frequency: Annotated[float, Parameter.Numeric(
        description="Minimum cutoff (Hz). Lower means more smoothing when stationary.",
        min=0.0,
    )]=0.5,
    beta: Annotated[float, Parameter.Numeric(
        description="Speed coefficient. Higher means less lag during fast motion.",
        min=0.0,
    )]=3.0,
    derivative_cutoff: Annotated[float, Parameter.Numeric(
        description="Cutoff (Hz) for the derivative low-pass.",
        min=0.0,
    )]=1.0,
) -> tuple[
    Annotated[ndarray, Parameter.Generic(description="Filtered data.")],
    Annotated[ndarray, Parameter.Generic(description="Updated internal state.")]
]:
    """
    Stateless 1-Euro filter step.
    """    
    # Unpack state
    n = (state.shape[0] - 1) // 2
    init_flag = float(state[0])
    x_prev = state[1 : 1 + n]
    dx_prev = state[1 + n : 1 + 2 * n]
    # Allocate new state (preserve dtype)
    new_state = state.copy()
    if init_flag < 0.5:
        # First call: seed with the raw sample, zero the derivative.
        x_filt = data.astype(state.dtype, copy=True)
        dx_filt = zeros(n, dtype=state.dtype)
    else:
        # Low-pass the derivative
        alpha_d = _alpha(delta_time, derivative_cutoff)
        dx_raw = (data - x_prev) / delta_time
        dx_filt = dx_prev + alpha_d * (dx_raw - dx_prev)
        # Adaptive per-channel cutoff, then low-pass the signal
        cutoff = cutoff_frequency + beta * abs(dx_filt)
        alpha = _alpha(delta_time, cutoff)
        x_filt = x_prev + alpha * (data - x_prev)
        x_filt = x_filt.astype(state.dtype, copy=False)
        dx_filt = dx_filt.astype(state.dtype, copy=False)
    # Update state
    new_state[0] = 1.0
    new_state[1 : 1 + n] = x_filt
    new_state[1 + n : 1 + 2 * n] = dx_filt
    # Return
    return x_filt, new_state

def _alpha(dt: float, cutoff: float | ndarray):
    """
    Smoothing factor for a first-order low-pass filter at the given cutoff.
    """
    r = 2.0 * pi * cutoff * dt
    return r / (r + 1.0)

if __name__ == "__main__":
    from numpy import array, float32
    from numpy.random import default_rng
    rng = default_rng(0)
    state = zeros(3, dtype=float32) # N=1
    print(f"{'raw':>8} {'filtered':>10}")
    for _ in range(8):
        sample = float32(1.0 + 0.3 * rng.standard_normal())
        filtered, state = one_euro_filter(array([sample]), state, delta_time=1/30)
        print(f"{sample:>8.3f} {float(filtered[0]):>10.3f}")