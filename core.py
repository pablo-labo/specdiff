from dataclasses import dataclass
import numpy as np

@dataclass
class ClientState:
    """
    Encapsulates the state for a single client.

    Attributes:
        hat_alpha: Effective acceptance rate estimate. Initialized to 0.5.
        X: Long-term goodput state. Initialized to 1.0.
        eta: Learning rate for alpha smoothing.
        beta: Learning rate for X smoothing.
    """
    eta: float
    beta: float
    hat_alpha: float = 0.5
    X: float = 1.0


def mu_func(S: int, alpha: float) -> float:
    """
    Calculates the expected goodput for a given draft length and acceptance rate.
    Formula: μ(S,α) = (1 - α^(S+1)) / (1 - α)
    [cite: 6]
    """
    # Handle numerical stability for α approaching 1.
    if alpha > 0.999:
        return float(S + 1)
    return (1 - alpha**(S + 1)) / (1 - alpha)


def update_alpha(hat_alpha_prev: float, tilda_alpha: float, eta: float) -> float:
    """
    Performs exponential smoothing to update the acceptance rate estimate.
    Formula: α̂ᵢ(t) = (1 - η)α̂ᵢ(t-1) + ηα̃ᵢ(t)
    [cite: 7]
    """
    return (1 - eta) * hat_alpha_prev + eta * tilda_alpha


def update_X(X_prev: float, x_realized: float, beta: float) -> float:
    """
    Updates the long-term performance variable (smoothed goodput).
    Formula: Xᵢ(t) = (1 - β)Xᵢ(t-1) + βxᵢ(t)
    [cite: 8]
    """
    return (1 - beta) * X_prev + beta * x_realized


def calculate_weight(X: float) -> float:
    """
    Calculates the gradient weight based on the utility function U(x) = log(x).
    Formula: wᵢ(t) = ∇Uᵢ(Xᵢ(t)) = 1 / Xᵢ(t)
    [cite: 10]
    """
    # Prevent division by zero or negative values.
    return 1 / np.maximum(X, 1e-6)
