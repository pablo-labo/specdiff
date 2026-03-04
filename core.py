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


def calculate_objective_weight(
    X: float, objective_mode: str = "paper_log_r", alpha: float = 1.0
) -> float:
    """
    Compute per-client gradient-like weight for scheduler objective.

    Modes:
        - paper_log_r:
            U(X) = log(X), dU/dX = 1 / X
        - r_plus_inv_alpha_aoi:
            U(X) = X + (1/alpha) * AoI, with AoI ~= 1 / X
            => U(X) = X + 1/(alpha*X)
            => dU/dX = 1 - 1/(alpha*X^2)
    """
    safe_x = float(np.maximum(X, 1e-6))
    safe_alpha = float(np.maximum(alpha, 1e-6))

    if objective_mode == "paper_log_r":
        return 1.0 / safe_x

    if objective_mode == "r_plus_inv_alpha_aoi":
        grad = 1.0 - 1.0 / (safe_alpha * safe_x * safe_x)
        # Negative gradient would imply that allocating more budget harms this
        # objective at current state; clamp for stable greedy allocation.
        return float(np.maximum(grad, 0.0))

    # Fallback to paper mode for unknown values.
    return 1.0 / safe_x
