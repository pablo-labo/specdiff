import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

        start_idx = self.current_prompt_ids.shape[1] - 1
        end_idx = start_idx + len(draft_tokens)
        
        target_preds = torch.argmax(logits[0, start_idx:end_idx], dim=-1)

        # --- 3. Compare ---
        accepted_len = 0
        accepted_tokens = []
        
        for i in range(len(draft_tokens)):
            if draft_tokens[i] == target_preds[i]:
                accepted_len += 1
                accepted_tokens.append(draft_tokens[i])
            else:
                accepted_tokens.append(target_preds[i]) 
                break
        
        return accepted_len, torch.tensor(accepted_tokens, device=self.device)

    def step_for_client(self, client_idx, draft_length_S):
        if not self.client_prompt_ids:
            raise ValueError("Prompts are not initialized. Call set_prompts first.")

        self.current_prompt_ids = self.client_prompt_ids[client_idx]
        accepted_len, accepted_tokens = self.step(draft_length_S)
        return accepted_len, accepted_tokens

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
