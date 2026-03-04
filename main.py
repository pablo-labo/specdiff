import numpy as np
import argparse
from core import ClientState, update_alpha, update_X
from scheduler import GradientScheduler
from engine import RealSpeculativeEngine

# --- 1. Simulation Environment Configuration ---
NUM_CLIENTS = 3  # Modified to 3 as requested (using 0.5B model)
TOTAL_SLOTS = 20 # Reduced slots for real inference demo
CLOUD_CAPACITY = 20
NUM_DRAFTS_K = 3
ETA = 0.1  # Learning rate for alpha smoothing
BETA = 0.1 # Learning rate for X smoothing
UPLOAD_DELAY_MS = [2.0, 2.0, 2.0]
FEEDBACK_DELAY_MS = [2.0, 2.0, 2.0]
OBJECTIVE_MODE = "paper_log_r"  # or "r_plus_inv_alpha_aoi"
OBJECTIVE_ALPHA = 1.0
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

def run_simulation(objective_mode: str = OBJECTIVE_MODE, objective_alpha: float = OBJECTIVE_ALPHA):
    """
    Main simulation entry point.
    """
    # --- Initialization ---
    scheduler = GradientScheduler(
        capacity=CLOUD_CAPACITY,
        objective_mode=objective_mode,
        objective_alpha=objective_alpha,
    )
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
    history_aoi_s = np.full((TOTAL_SLOTS, NUM_CLIENTS), np.nan)

    print(
        "Starting simulation: "
        f"N={NUM_CLIENTS}, T={TOTAL_SLOTS}, C={CLOUD_CAPACITY}, "
        f"objective={objective_mode}, alpha={objective_alpha}"
    )
    print("-" * 60)

    # --- 2. Simulation Loop ---
    for t in range(TOTAL_SLOTS):
        # 1. Scheduling Phase [cite: 11]
        S_allocations = scheduler.allocate(clients)
        
        total_goodput_in_slot = 0
        # 2. Real Inference Step
        # Draft per client, then verify all drafts in one target-model batch.
        client_results, draft_times, verify_time_s = engine.step_all_clients_timed(S_allocations)
        client_step_outputs = []
        for i in range(NUM_CLIENTS):
            S_i, l_i, new_tokens = client_results[i]
            engine.update_prompt_for_client(client_idx=i, new_tokens=new_tokens)
            client_step_outputs.append((S_i, l_i, len(new_tokens)))

        # Time model aligned with SpecDiff Eq. (14)(16)(17)(18):
        # W(t) = max_i(G_i + U_i), V(t) = verifier time, F(t) = max_i(D_i)
        upload_delays_s = [ms / 1000.0 for ms in UPLOAD_DELAY_MS[:NUM_CLIENTS]]
        feedback_delays_s = [ms / 1000.0 for ms in FEEDBACK_DELAY_MS[:NUM_CLIENTS]]
        while len(upload_delays_s) < NUM_CLIENTS:
            upload_delays_s.append(0.0)
        while len(feedback_delays_s) < NUM_CLIENTS:
            feedback_delays_s.append(0.0)

        barrier_wait_s = max(
            (draft_times[i] + upload_delays_s[i] for i in range(NUM_CLIENTS)),
            default=0.0,
        )
        # Verification stage uses one shared batch forward in target model.
        verify_stage_s = verify_time_s
        feedback_stage_s = max(feedback_delays_s, default=0.0)
        slot_time_s = barrier_wait_s + verify_stage_s + feedback_stage_s
        slot_time_s = max(slot_time_s, 1e-6)

        for i in range(NUM_CLIENTS):
            S_i, l_i, _ = client_step_outputs[i]
            # Paper-aligned rate proxy: R_i(t) = A_i(t) / T(t), with A_i ~= l_i.
            x_i = l_i / slot_time_s

            # 4. State Update [cite: 131]
            tilda_alpha = l_i / S_i if S_i > 0 else 0.0
            clients[i].hat_alpha = update_alpha(clients[i].hat_alpha, tilda_alpha, clients[i].eta)
            clients[i].X = update_X(clients[i].X, x_i, clients[i].beta)

            history_allocations[t, i] = S_i
            history_goodputs[t, i] = x_i
            if l_i > 0:
                history_aoi_s[t, i] = slot_time_s / l_i
            total_goodput_in_slot += x_i

        # --- Data Monitoring ---
        utilization = np.sum(S_allocations) / CLOUD_CAPACITY
        print(
            f"Slot {t+1: >3}/{TOTAL_SLOTS} | Total Goodput: {total_goodput_in_slot: >8.2f} "
            f"| SlotTime(W+V+F): {slot_time_s*1000: >7.1f} ms | Utilization: {utilization: >4.1%}"
        )

    # --- 3. Final Report ---
    print("\n" + "=" * 60)
    print("Simulation Finished: Summary Report")
    print("=" * 60)
    print(
        f"{'Client': >8} | {'True α': >8} | {'Avg S': >8} | "
        f"{'Avg R': >8} | {'Avg AoI(s)': >10} | {'Fairness (R/S)': >15}"
    )
    print("-" * 60)

    avg_allocations = np.mean(history_allocations, axis=0)
    avg_goodputs = np.mean(history_goodputs, axis=0)
    avg_aoi_s = np.nanmean(history_aoi_s, axis=0)
    avg_aoi_s = np.where(np.isnan(avg_aoi_s), 0.0, avg_aoi_s)

    for i in range(NUM_CLIENTS):
        # Avoid division by zero for fairness metric if a client got no allocation.
        fairness = avg_goodputs[i] / avg_allocations[i] if avg_allocations[i] > 0 else 0
        print(
            f"{i: >8} | {'N/A': >8} | {avg_allocations[i]: >8.2f} | "
            f"{avg_goodputs[i]: >8.2f} | {avg_aoi_s[i]: >10.3f} | {fairness: >15.2f}"
        )
    print("-" * 60)
    
    print("\nFinal Generated Texts:")
    for i, text in enumerate(engine.decode_all()):
        print(f"[Client {i}] {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speculative scheduling simulation.")
    parser.add_argument(
        "--objective-mode",
        default=OBJECTIVE_MODE,
        choices=["paper_log_r", "r_plus_inv_alpha_aoi"],
        help="Scheduling objective mode.",
    )
    parser.add_argument(
        "--objective-alpha",
        type=float,
        default=OBJECTIVE_ALPHA,
        help="Alpha coefficient used by r_plus_inv_alpha_aoi mode.",
    )
    args = parser.parse_args()
    run_simulation(objective_mode=args.objective_mode, objective_alpha=args.objective_alpha)
