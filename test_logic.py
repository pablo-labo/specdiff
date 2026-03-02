import pytest
import numpy as np
from core import ClientState, mu_func, update_alpha
from scheduler import GradientScheduler

#
# 1. 数值稳定性测试 (Boundary Conditions)
#
def test_mu_func_stability():
    """
    Validates the numerical stability and boundary conditions of mu_func.
    Formula: μ(S,α) = (1 - α^(S+1)) / (1 - α) [cite: 6]
    """
    S = 5
    # Test case: alpha = 0. The sum of a geometric series with r=0 is just the first term a=1.
    assert mu_func(S, alpha=0.0) == 1.0, "Failure: alpha=0 should yield 1.0"
    
    # Test case: alpha -> 1. The function should use the limit value S + 1.
    assert mu_func(S, alpha=1.0) == S + 1, "Failure: alpha=1.0 should yield S+1"
    assert mu_func(S, alpha=0.9999) == S + 1, "Failure: alpha->1 should yield S+1"

#
# 2. 调度算法一致性测试 (Resource Constraint)
#
def test_scheduler_consistency():
    """
    Validates that the GradientScheduler respects the resource capacity constraint.
    Constraint: Σ Sᵢ ≤ C [cite: 9]
    """
    CAPACITY = 10
    clients = [
        ClientState(eta=0.1, beta=0.1, hat_alpha=0.5, X=1.0),
        ClientState(eta=0.1, beta=0.1, hat_alpha=0.6, X=1.2),
        ClientState(eta=0.1, beta=0.1, hat_alpha=0.7, X=0.8),
    ]
    
    scheduler = GradientScheduler(capacity=CAPACITY)
    allocations = scheduler.allocate(clients)
    
    # Verification: sum of allocations must not exceed capacity.
    assert sum(allocations) <= CAPACITY, f"Failure: sum(S)={sum(allocations)} > C={CAPACITY}"
    
    # Verification: allocations must be non-negative integers.
    assert all(isinstance(s, int) and s >= 0 for s in allocations), "Failure: S must contain non-negative integers"

#
# 3. 公平性补偿逻辑验证 (Incentive Test)
#
def test_scheduler_fairness_incentive():
    """
    Validates that the scheduler prioritizes clients with poor long-term performance (low X)
    due to the gradient weight w = 1/X.
    Formula: wᵢ(t) = ∇Uᵢ(Xᵢ(t)) [cite: 10]
    """
    CAPACITY = 10
    
    # Client A: Poor performance (low X), thus high weight.
    client_a = ClientState(eta=0.1, beta=0.1, hat_alpha=0.2, X=1e-4) # w ≈ 10000
    
    # Client B: Good performance (high X), thus low weight.
    client_b = ClientState(eta=0.1, beta=0.1, hat_alpha=0.9, X=100.0) # w = 0.01
    
    clients = [client_a, client_b]
    
    scheduler = GradientScheduler(capacity=CAPACITY)
    allocations = scheduler.allocate(clients)
    
    # Verification: Client A should receive more (or all) resources despite lower alpha.
    assert allocations[0] > allocations[1], "Failure: Scheduler did not prioritize client with higher gradient weight"

#
# 4. 状态演化测试 (Smoothing)
#
def test_smoothing_convergence():
    """
    Validates the exponential smoothing logic for state evolution.
    Formula: α̂ᵢ(t) = (1 - η)α̂ᵢ(t-1) + ηα̃ᵢ(t) [cite: 7]
    """
    ETA = 0.1
    hat_alpha_initial = 0.0
    tilda_alpha_observed = 1.0 # Constant observation
    
    hat_alpha_current = hat_alpha_initial
    num_steps = 10
    for _ in range(num_steps):
        hat_alpha_current = update_alpha(hat_alpha_current, tilda_alpha_observed, ETA)
        
    # After k steps, the closed-form solution for EMA is:
    # E_k = (1 - (1-η)^k) * V + (1-η)^k * E_0
    expected_alpha = (1 - (1 - ETA)**num_steps) * tilda_alpha_observed + ((1 - ETA)**num_steps) * hat_alpha_initial
    
    # Verification: The final value must be close to the analytically expected value.
    assert np.isclose(hat_alpha_current, expected_alpha), "Failure: Smoothing did not converge to expected value"
    
    # Verification: The value must have moved from initial towards observed.
    initial_dist = abs(hat_alpha_initial - tilda_alpha_observed)
    final_dist = abs(hat_alpha_current - tilda_alpha_observed)
    assert final_dist < initial_dist, "Failure: Smoothing did not reduce distance to observed value"
