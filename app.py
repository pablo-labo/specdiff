import http.server
import socketserver
import json
import numpy as np
from dataclasses import dataclass
import threading

# --- 1. Core Mathematical Model (NumPy Optimized) ---

@dataclass
class ClientState:
    """
    Encapsulates all state for a single client.
    """
    eta: float
    beta: float
    hat_alpha: float = 0.5
    X: float = 1.0

def mu_func(S: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Calculates expected goodput using vectorized operations.
    Formula: μ(S,α) = (1 - α^(S+1)) / (1 - α) [cite: 6]
    """
    # Use np.isclose for robust floating point comparison.
    close_to_one = np.isclose(alpha, 1.0)
    # Initialize result array
    result = np.zeros_like(alpha, dtype=float)
    
    # Case 1: alpha is close to 1, use limit S + 1
    result[close_to_one] = S[close_to_one] + 1.0
    
    # Case 2: alpha is not close to 1, use standard formula
    not_close_to_one = ~close_to_one
    result[not_close_to_one] = (1 - alpha[not_close_to_one]**(S[not_close_to_one] + 1)) / (1 - alpha[not_close_to_one])
    
    return result

def calculate_weight(X: np.ndarray) -> np.ndarray:
    """
    Calculates gradient weight. wᵢ(t) = ∇Uᵢ(Xᵢ(t)) = 1/Xᵢ(t) [cite: 10]
    """
    return 1.0 / np.maximum(X, 1e-6)

class GradientScheduler:
    """
    Implements the gradient-based scheduling algorithm.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity

    def allocate(self, clients: list[ClientState]) -> np.ndarray:
        """
        Allocates verification budget using a vectorized greedy algorithm.
        Objective: arg max_S Σ wᵢ * μ(Sᵢ, α̂ᵢ) [cite: 11]
        """
        num_clients = len(clients)
        allocations = np.zeros(num_clients, dtype=int)
        
        weights = calculate_weight(np.array([c.X for c in clients]))
        alphas = np.array([c.hat_alpha for c in clients])

        for _ in range(self.capacity):
            current_mu = mu_func(allocations, alphas)
            next_mu = mu_func(allocations + 1, alphas)
            
            # Gainᵢ = wᵢ * [μ(Sᵢ+1, α̂ᵢ) - μ(Sᵢ, α̂ᵢ)]
            marginal_gains = weights * (next_mu - current_mu)
            
            if np.sum(marginal_gains) <= 0:
                break 
            
            best_client_idx = np.argmax(marginal_gains)
            allocations[best_client_idx] += 1
            
        return allocations

# --- 2. Backend Simulation Engine ---

def simulate_best_of_k(draft_length: int, true_alpha: float, k: int) -> int:
    """
    Simulates the best-of-K edge generation process. [cite: 39]
    """
    if draft_length == 0:
        return 0
    # Generate K draft outcomes at once
    draft_accepts = np.random.binomial(1, true_alpha, size=(k, draft_length))
    # Find first rejection (index of first 0) for each draft
    # Note: np.argmin works because it finds the first minimum (0)
    accepted_lengths = np.argmin(draft_accepts, axis=1)
    # If a row has no zeros, argmin returns 0. Correct this case.
    all_accepted_mask = (draft_accepts.all(axis=1))
    accepted_lengths[all_accepted_mask] = draft_length
    
    return np.max(accepted_lengths)

def run_simulation(num_slots=50):
    """
    Executes the full simulation and returns historical data.
    """
    NUM_CLIENTS = 5
    CLOUD_CAPACITY = 20
    NUM_DRAFTS_K = 3
    ETA = 0.1
    BETA = 0.1
    
    scheduler = GradientScheduler(capacity=CLOUD_CAPACITY)
    clients = [ClientState(eta=ETA, beta=BETA) for _ in range(NUM_CLIENTS)]
    true_alphas = np.random.uniform(0.1, 0.9, size=NUM_CLIENTS)

    total_goodput_history = []
    allocations_history = []

    for _ in range(num_slots):
        # 1. Scheduling [cite: 11, 12]
        S_allocations = scheduler.allocate(clients)
        allocations_history.append(S_allocations)

        slot_total_goodput = 0
        for i, client in enumerate(clients):
            S_i = S_allocations[i]
            
            # 2. Simulation (Generation & Verification) [cite: 39, 50]
            l_i = simulate_best_of_k(S_i, true_alphas[i], NUM_DRAFTS_K)
            x_i = 1 + l_i  # [cite: 53]
            
            # 3. State Update [cite: 13]
            tilda_alpha = l_i / S_i if S_i > 0 else 0.0
            client.hat_alpha = (1 - client.eta) * client.hat_alpha + client.eta * tilda_alpha # [cite: 7]
            client.X = (1 - client.beta) * client.X + client.beta * x_i # [cite: 8]
            
            slot_total_goodput += x_i
        
        total_goodput_history.append(slot_total_goodput)

    final_allocations = np.mean(allocations_history, axis=0)
    
    return {
        "total_goodput_history": [int(x) for x in total_goodput_history],
        "final_allocations": final_allocations.tolist(),
        "client_labels": [f"Client {i} (α={true_alphas[i]:.2f})" for i in range(NUM_CLIENTS)],
    }

# --- 3. Lightweight Web Server & 4. Frontend UI ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Speculative Decoding Simulation</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; background: #f0f2f5; margin: 0; }
        .container { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-top: 2rem; width: 90%; max-width: 1200px; }
        h1 { text-align: center; color: #333; }
        button { display: block; margin: 1rem auto; padding: 0.75rem 1.5rem; font-size: 1rem; color: white; background-color: #007bff; border: none; border-radius: 5px; cursor: pointer; transition: background-color 0.3s; }
        button:hover { background-color: #0056b3; }
        .charts { display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 2rem; }
        .chart-container { width: 48%; min-width: 300px; margin-bottom: 2rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Edge Diffusion Speculative Decoding</h1>
        <button id="run-sim-btn">Run Simulation</button>
        <div class="charts">
            <div class="chart-container">
                <canvas id="goodputChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="allocationChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const goodputCtx = document.getElementById('goodputChart').getContext('2d');
        const allocationCtx = document.getElementById('allocationChart').getContext('2d');
        let goodputChart, allocationChart;

        function createCharts(labels) {
            if (goodputChart) goodputChart.destroy();
            goodputChart = new Chart(goodputCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Total System Goodput per Slot',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        fill: false,
                    }]
                }
            });

            if (allocationChart) allocationChart.destroy();
            allocationChart = new Chart(allocationCtx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Average Resource Allocation (Sᵢ)',
                        data: [],
                        backgroundColor: 'rgba(153, 102, 255, 0.6)',
                    }]
                },
                options: { indexAxis: 'y' }
            });
        }
        
        document.getElementById('run-sim-btn').addEventListener('click', () => {
            const btn = document.getElementById('run-sim-btn');
            btn.textContent = 'Running...';
            btn.disabled = true;

            fetch('/api/run')
                .then(response => response.json())
                .then(data => {
                    createCharts(data.client_labels); // Re-create charts with new labels

                    goodputChart.data.labels = Array.from({length: data.total_goodput_history.length}, (_, i) => i + 1);
                    goodputChart.data.datasets[0].data = data.total_goodput_history;
                    goodputChart.update();

                    allocationChart.data.datasets[0].data = data.final_allocations;
                    allocationChart.update();
                })
                .finally(() => {
                    btn.textContent = 'Run Simulation';
                    btn.disabled = false;
                });
        });
        
        // Initial empty charts
        createCharts([]);
    </script>
</body>
</html>
"""

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif self.path == '/api/run':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            sim_data = run_simulation()
            self.wfile.write(json.dumps(sim_data).encode('utf-8'))
        else:
            self.send_error(404, "File Not Found")

# --- 5. Server Startup ---
def run_server(port=8000):
    """
    Starts the HTTP server.
    """
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving at http://localhost:{port}")
        # Open the web browser automatically
        threading.Timer(1, lambda: __import__('webbrowser').open(f'http://localhost:{port}')).start()
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
