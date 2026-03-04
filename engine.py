import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
from torch.nn.utils.rnn import pad_sequence


def _sample_residual_token(p_probs: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
    """
    Sample one token from the normalized residual distribution:
        r(x) ∝ max(0, p(x) - q(x))
    If the residual mass is numerically zero, fall back to sampling from p.
    """
    residual = torch.clamp(p_probs - q_probs, min=0.0)
    residual_mass = residual.sum()
    if residual_mass <= 1e-12:
        return torch.multinomial(p_probs, num_samples=1).squeeze(0)
    residual = residual / residual_mass
    return torch.multinomial(residual, num_samples=1).squeeze(0)

class RealSpeculativeEngine:
    def __init__(self, target_model_name, draft_model_name, device="cuda"):
        print(f"Loading models on {device} with 4-bit quantization...")
        
        # 4-bit Quantization Config to save VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # 1. Load Tokenizer (Assuming they share the same tokenizer, which Qwen2.5 does)
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.device = device

        # 2. Load Target Model (Verification Server) - 7B
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name, 
            quantization_config=bnb_config,
            device_map="auto" 
        )
        
        # 3. Load Draft Model (Edge Drafter) - 0.5B
        # We load it once, but can use it multiple times to simulate multiple clients
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name, 
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.current_prompt_ids = None
        self.client_prompt_ids = []

    def set_prompt(self, text):
        self.current_prompt_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        self.client_prompt_ids = [self.current_prompt_ids.clone()]

    def set_prompts(self, texts):
        self.client_prompt_ids = [
            self.tokenizer.encode(text, return_tensors="pt").to(self.device) for text in texts
        ]
        self.current_prompt_ids = self.client_prompt_ids[0].clone()

    def get_prompt_ids_clone(self, client_idx):
        return self.client_prompt_ids[client_idx].clone()

    def step(self, draft_length_S):
        """
        Executes one step of speculative decoding:
        1. Draft generates S tokens.
        2. Target verifies them.
        Returns: (accepted_length, accepted_token_ids)
        """
        if self.current_prompt_ids is None:
            raise ValueError("Prompt is not initialized. Call set_prompt first.")

        if draft_length_S == 0:
            return 0, torch.tensor([], device=self.device)

        # --- 1. Draft Generation ---
        draft_outputs = self.draft_model.generate(
            self.current_prompt_ids,
            max_new_tokens=draft_length_S,
            do_sample=False, 
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        draft_tokens = draft_outputs[0, self.current_prompt_ids.shape[1]:]

        # --- 2. Verification ---
        with torch.no_grad():
            target_outputs = self.target_model(draft_outputs)
            logits = target_outputs.logits
            draft_logits = self.draft_model(draft_outputs).logits

        start_idx = self.current_prompt_ids.shape[1] - 1
        end_idx = start_idx + len(draft_tokens)

        p_logits = logits[0, start_idx:end_idx]
        q_logits = draft_logits[0, start_idx:end_idx]
        p_probs_all = torch.softmax(p_logits, dim=-1)
        q_probs_all = torch.softmax(q_logits, dim=-1)

        # --- 3. Compare ---
        accepted_len = 0
        accepted_tokens = []

        for i in range(len(draft_tokens)):
            token_id = draft_tokens[i]
            p_probs = p_probs_all[i]
            q_probs = q_probs_all[i]

            p_token = p_probs[token_id]
            q_token = torch.clamp(q_probs[token_id], min=1e-12)
            accept_prob = torch.clamp(p_token / q_token, max=1.0)

            if torch.rand((), device=self.device) < accept_prob:
                accepted_len += 1
                accepted_tokens.append(token_id)
            else:
                replacement = _sample_residual_token(p_probs, q_probs)
                accepted_tokens.append(replacement)
                break

        return accepted_len, torch.tensor(accepted_tokens, device=self.device)

    def step_timed(self, draft_length_S):
        """
        Same as step(), but returns timing breakdown:
        (accepted_len, accepted_tokens, draft_time_s, verify_time_s)
        """
        if self.current_prompt_ids is None:
            raise ValueError("Prompt is not initialized. Call set_prompt first.")

        if draft_length_S == 0:
            return 0, torch.tensor([], device=self.device), 0.0, 0.0

        draft_start = time.perf_counter()
        draft_outputs = self.draft_model.generate(
            self.current_prompt_ids,
            max_new_tokens=draft_length_S,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        draft_time_s = time.perf_counter() - draft_start
        draft_tokens = draft_outputs[0, self.current_prompt_ids.shape[1]:]

        verify_start = time.perf_counter()
        with torch.no_grad():
            target_outputs = self.target_model(draft_outputs)
            logits = target_outputs.logits
            draft_logits = self.draft_model(draft_outputs).logits

        start_idx = self.current_prompt_ids.shape[1] - 1
        end_idx = start_idx + len(draft_tokens)

        p_logits = logits[0, start_idx:end_idx]
        q_logits = draft_logits[0, start_idx:end_idx]
        p_probs_all = torch.softmax(p_logits, dim=-1)
        q_probs_all = torch.softmax(q_logits, dim=-1)

        accepted_len = 0
        accepted_tokens = []
        for i in range(len(draft_tokens)):
            token_id = draft_tokens[i]
            p_probs = p_probs_all[i]
            q_probs = q_probs_all[i]

            p_token = p_probs[token_id]
            q_token = torch.clamp(q_probs[token_id], min=1e-12)
            accept_prob = torch.clamp(p_token / q_token, max=1.0)

            if torch.rand((), device=self.device) < accept_prob:
                accepted_len += 1
                accepted_tokens.append(token_id)
            else:
                replacement = _sample_residual_token(p_probs, q_probs)
                accepted_tokens.append(replacement)
                break
        verify_time_s = time.perf_counter() - verify_start

        return accepted_len, torch.tensor(accepted_tokens, device=self.device), draft_time_s, verify_time_s

    def step_for_client(self, client_idx, draft_length_S):
        if not self.client_prompt_ids:
            raise ValueError("Prompts are not initialized. Call set_prompts first.")

        self.current_prompt_ids = self.client_prompt_ids[client_idx]
        accepted_len, accepted_tokens = self.step(draft_length_S)
        return accepted_len, accepted_tokens

    def step_for_client_timed(self, client_idx, draft_length_S):
        if not self.client_prompt_ids:
            raise ValueError("Prompts are not initialized. Call set_prompts first.")

        self.current_prompt_ids = self.client_prompt_ids[client_idx]
        accepted_len, accepted_tokens, draft_time_s, verify_time_s = self.step_timed(draft_length_S)
        return accepted_len, accepted_tokens, draft_time_s, verify_time_s

    def step_all_clients_timed(self, draft_lengths):
        """
        Draft per client, then verify all drafted sequences in one target-model batch.
        Returns:
            (client_results, draft_times, verify_time_s)
        where:
            client_results[i] = (S_i, accepted_len_i, accepted_tokens_i)
        """
        if not self.client_prompt_ids:
            raise ValueError("Prompts are not initialized. Call set_prompts first.")
        if len(draft_lengths) != len(self.client_prompt_ids):
            raise ValueError("draft_lengths size must match number of clients.")

        num_clients = len(self.client_prompt_ids)
        draft_times = [0.0 for _ in range(num_clients)]
        accepted_lens = [0 for _ in range(num_clients)]
        accepted_tokens_all = [
            torch.tensor([], device=self.device, dtype=torch.long) for _ in range(num_clients)
        ]

        active_client_indices = []
        active_full_sequences = []
        active_draft_tokens = []
        active_prompt_lens = []
        active_q_probs_all = []

        # Draft stage (simulated per-client drafter).
        for i in range(num_clients):
            S_i = int(draft_lengths[i])
            if S_i <= 0:
                continue

            prompt_ids = self.client_prompt_ids[i]
            draft_start = time.perf_counter()
            draft_outputs = self.draft_model.generate(
                prompt_ids,
                max_new_tokens=S_i,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            draft_times[i] = time.perf_counter() - draft_start

            prompt_len = prompt_ids.shape[1]
            draft_tokens = draft_outputs[0, prompt_len:]
            if draft_tokens.numel() == 0:
                continue

            with torch.no_grad():
                draft_logits = self.draft_model(draft_outputs).logits
            q_logits = draft_logits[0, prompt_len - 1 : prompt_len - 1 + draft_tokens.shape[0]]
            q_probs_all = torch.softmax(q_logits, dim=-1)

            active_client_indices.append(i)
            active_full_sequences.append(draft_outputs[0])
            active_draft_tokens.append(draft_tokens)
            active_prompt_lens.append(prompt_len)
            active_q_probs_all.append(q_probs_all)

        verify_time_s = 0.0
        if active_client_indices:
            pad_token_id = self.tokenizer.eos_token_id
            if pad_token_id is None:
                pad_token_id = 0

            padded_inputs = pad_sequence(
                active_full_sequences, batch_first=True, padding_value=pad_token_id
            )
            attention_mask = torch.zeros_like(padded_inputs, dtype=torch.long)
            for row, seq in enumerate(active_full_sequences):
                attention_mask[row, : seq.shape[0]] = 1

            verify_start = time.perf_counter()
            with torch.no_grad():
                target_outputs = self.target_model(
                    input_ids=padded_inputs, attention_mask=attention_mask
                )
                target_logits = target_outputs.logits
            verify_time_s = time.perf_counter() - verify_start

            # Per-client acceptance / residual correction on top of shared batch logits.
            for row, client_idx in enumerate(active_client_indices):
                draft_tokens = active_draft_tokens[row]
                prompt_len = active_prompt_lens[row]
                p_logits = target_logits[row, prompt_len - 1 : prompt_len - 1 + draft_tokens.shape[0]]
                p_probs_all = torch.softmax(p_logits, dim=-1)
                q_probs_all = active_q_probs_all[row]

                accepted_len = 0
                accepted_tokens = []
                for pos in range(draft_tokens.shape[0]):
                    token_id = draft_tokens[pos]
                    p_probs = p_probs_all[pos]
                    q_probs = q_probs_all[pos]

                    p_token = p_probs[token_id]
                    q_token = torch.clamp(q_probs[token_id], min=1e-12)
                    accept_prob = torch.clamp(p_token / q_token, max=1.0)

                    if torch.rand((), device=self.device) < accept_prob:
                        accepted_len += 1
                        accepted_tokens.append(token_id)
                    else:
                        replacement = _sample_residual_token(p_probs, q_probs)
                        accepted_tokens.append(replacement)
                        break

                accepted_lens[client_idx] = accepted_len
                accepted_tokens_all[client_idx] = torch.tensor(
                    accepted_tokens, device=self.device, dtype=torch.long
                )

        client_results = []
        for i in range(num_clients):
            client_results.append((int(draft_lengths[i]), accepted_lens[i], accepted_tokens_all[i]))
        return client_results, draft_times, verify_time_s

    def update_prompt(self, new_tokens):
        if len(new_tokens) > 0:
            self.current_prompt_ids = torch.cat([self.current_prompt_ids, new_tokens.unsqueeze(0)], dim=1)

    def update_prompt_for_client(self, client_idx, new_tokens):
        if not self.client_prompt_ids:
            raise ValueError("Prompts are not initialized. Call set_prompts first.")

        self.current_prompt_ids = self.client_prompt_ids[client_idx]
        self.update_prompt(new_tokens)
        self.client_prompt_ids[client_idx] = self.current_prompt_ids

    def decode(self):
        return self.tokenizer.decode(self.current_prompt_ids[0], skip_special_tokens=True)

    def decode_client(self, client_idx):
        return self.tokenizer.decode(self.client_prompt_ids[client_idx][0], skip_special_tokens=True)

    def decode_all(self):
        return [self.decode_client(i) for i in range(len(self.client_prompt_ids))]
