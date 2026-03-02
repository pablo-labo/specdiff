import numpy as np
from typing import List
from core import ClientState, mu_func, calculate_weight

class GradientScheduler:
    """
    Implements the gradient-based scheduling algorithm to allocate verification budget.
    """

    def __init__(self, capacity: int):
        """
        Initializes the scheduler with a total verification capacity.

        Args:
            capacity: The total verification budget C. [cite: 9]
        """
        self.capacity = capacity

    def allocate(self, clients: List[ClientState]) -> List[int]:
        """
        Allocates the verification budget among clients using a greedy approach.
        The allocation aims to solve: arg max_S Σ wᵢ * μ(Sᵢ, α̂ᵢ)
        subject to: Σ Sᵢ ≤ C.
        [cite: 11]

        Args:
            clients: A list of ClientState objects.

        Returns:
            A list containing the allocated draft length Sᵢ for each client.
        """
        num_clients = len(clients)
        allocations = [0] * num_clients

        weights = [calculate_weight(client.X) for client in clients]
        alphas = [client.hat_alpha for client in clients]

        # Greedy allocation loop for C iterations.
        for _ in range(self.capacity):
            marginal_gains = np.zeros(num_clients)
            for i in range(num_clients):
                # Calculate marginal gain for incrementing Sᵢ by 1.
                # Gainᵢ = wᵢ * [μ(Sᵢ+1, α̂ᵢ) - μ(Sᵢ, α̂ᵢ)]
                current_mu = mu_func(allocations[i], alphas[i])
                next_mu = mu_func(allocations[i] + 1, alphas[i])
                marginal_gains[i] = weights[i] * (next_mu - current_mu)

            # Allocate budget to the client with the highest marginal gain.
            if np.sum(marginal_gains) > 0:
                best_client_idx = np.argmax(marginal_gains)
                allocations[best_client_idx] += 1
            else:
                # No positive gain available, stop allocation.
                break
        
        return allocations
