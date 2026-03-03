import numpy as np
from core import ClientState, update_alpha, update_X
from scheduler import GradientScheduler
from engine import RealSpeculativeEngine
import time

# --- 1. Simulation Environment Configuration ---
NUM_CLIENTS = 3  # Modified to 3 as requested (using 0.5B model)
TOTAL_SLOTS = 20 # Reduced slots for real inference demo
CLOUD_CAPACITY = 20
NUM_DRAFTS_K = 3
ETA = 0.1  # Learning rate for alpha smoothing
BETA = 0.1 # Learning rate for X smoothing
INITIAL_PROMPTS = [
    "Write a concise summary of why speculative decoding improves inference throughput.",
    "Draft a short explanation of fairness-aware scheduling in multi-client model serving.",
    "Explain the tradeoff between draft length and acceptance rate in speculative decoding.",
]

def simulate_best_of_k_selection(draft_length: int, true_alpha: float, k: int) -> int:
    """
    Simulates the best-of-K edge generation process.
    For K candidates, the one with the longest accepted prefix is chosen. [cite: 39]

    Returns:
        The length of the accepted prefix (l_i) for the chosen draft.
    """
    if draft_length == 0:
        return 0

    max_accepted_len = 0
    for _ in range(k):
        # Simulate a draft verification by drawing from a Bernoulli distribution.
        draft_accepts = np.random.binomial(1, true_alpha, size=draft_length)
        
        # Find the length of the continuous accepted prefix.
        first_rejection = np.where(draft_accepts == 0)[0]
        if len(first_rejection) > 0:
            accepted_len = first_rejection[0]
        else:
            accepted_len = draft_length # All tokens were accepted
        
        if accepted_len > max_accepted_len:
            max_accepted_len = accepted_len
            
    return max_accepted_len

def run_simulation():
    """
    Main simulation entry point.
    """
    # --- Initialization ---
    scheduler = GradientScheduler(capacity=CLOUD_CAPACITY)
    clients = [ClientState(eta=ETA, beta=BETA) for _ in range(NUM_CLIENTS)]
    
    # --- Initialize Real Engine ---
    print("Initializing AI Models (Qwen2.5-7B & 0.5B)...")
    # Ensure you have installed: pip install transformers accelerate bitsandbytes
    engine = RealSpeculativeEngine("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct")
    if len(INITIAL_PROMPTS) != NUM_CLIENTS:
        raise ValueError("INITIAL_PROMPTS length must match NUM_CLIENTS.")
    engine.set_prompts(INITIAL_PROMPTS)
    print("Initial Prompts:")
    for i, prompt in enumerate(INITIAL_PROMPTS):
        print(f"  Client {i}: {prompt}")

    # Data stores for metrics
    history_allocations = np.zeros((TOTAL_SLOTS, NUM_CLIENTS))
    history_goodputs = np.zeros((TOTAL_SLOTS, NUM_CLIENTS))

    print(f"Starting simulation: N={NUM_CLIENTS}, T={TOTAL_SLOTS}, C={CLOUD_CAPACITY}")
    print("-" * 60)

    # --- 2. Simulation Loop ---
    for t in range(TOTAL_SLOTS):
        # 1. Scheduling Phase [cite: 11]
        S_allocations = scheduler.allocate(clients)
        
        total_goodput_in_slot = 0
        client_step_outputs = []
        client_step_latencies = []
        
        for i in range(NUM_CLIENTS):
            S_i = S_allocations[i]
            
            # 2. Real Inference Step
            # Each client evolves its own independent prompt/context.
            step_start = time.perf_counter()
            l_i, new_tokens = engine.step_for_client(client_idx=i, draft_length_S=S_i)
            elapsed = time.perf_counter() - step_start
            
            # Update only this client's prompt.
            engine.update_prompt_for_client(client_idx=i, new_tokens=new_tokens)
            client_step_outputs.append((S_i, l_i, len(new_tokens)))
            client_step_latencies.append(elapsed)

        # Simulated parallel slot duration uses the slowest client in the slot.
        slot_time_s = max(client_step_latencies) if client_step_latencies else 1e-6
        slot_time_s = max(slot_time_s, 1e-6)

        for i in range(NUM_CLIENTS):
            S_i, l_i, _ = client_step_outputs[i]
            # Time-aware goodput under parallel-slot approximation.
            x_i = (1 + l_i) / slot_time_s

            # 4. State Update [cite: 131]
            tilda_alpha = l_i / S_i if S_i > 0 else 0.0
            clients[i].hat_alpha = update_alpha(clients[i].hat_alpha, tilda_alpha, clients[i].eta)
            clients[i].X = update_X(clients[i].X, x_i, clients[i].beta)

            history_allocations[t, i] = S_i
            history_goodputs[t, i] = x_i
            total_goodput_in_slot += x_i

        # --- Data Monitoring ---
        utilization = np.sum(S_allocations) / CLOUD_CAPACITY
        print(
            f"Slot {t+1: >3}/{TOTAL_SLOTS} | Total Goodput: {total_goodput_in_slot: >8.2f} "
            f"| SlotTime(max): {slot_time_s*1000: >7.1f} ms | Utilization: {utilization: >4.1%}"
        )

    # --- 3. Final Report ---
    print("\n" + "=" * 60)
    print("Simulation Finished: Summary Report")
    print("=" * 60)
    print(f"{'Client': >8} | {'True α': >8} | {'Avg S': >8} | {'Avg x': >8} | {'Fairness (x/S)': >15}")
    print("-" * 60)

    avg_allocations = np.mean(history_allocations, axis=0)
    avg_goodputs = np.mean(history_goodputs, axis=0)

    for i in range(NUM_CLIENTS):
        # Avoid division by zero for fairness metric if a client got no allocation.
        fairness = avg_goodputs[i] / avg_allocations[i] if avg_allocations[i] > 0 else 0
        print(f"{i: >8} | {'N/A': >8} | {avg_allocations[i]: >8.2f} | {avg_goodputs[i]: >8.2f} | {fairness: >15.2f}")
    print("-" * 60)
    
    print("\nFinal Generated Texts:")
    for i, text in enumerate(engine.decode_all()):
        print(f"[Client {i}] {text}")


if __name__ == "__main__":
    run_simulation()
