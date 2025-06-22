import argparse
import sys
import glob
import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from safetensors.torch import load_file
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv

# ==============================================================================
# Two-Pass Attention Implementation (from train.py)
# ==============================================================================
class TwoPassAttention(Qwen2Attention):
    """
    Implements a two-pass attention mechanism where the first pass scouts for token relationships
    and the second pass uses this information to gate the Q, K, and V projections.
    """
    def __init__(self, config, lm_head, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.lm_head = lm_head

        # Add attributes for analysis
        self.analyze_weights = False
        self.gate_magnitudes = []

        gate_input_dim = config.hidden_size + 1
        self.gate_scale = 0.2
        self.q_uncertainty_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 16, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(16, config.hidden_size, dtype=torch.float32)
        )
        self.k_uncertainty_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 16, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(16, config.hidden_size, dtype=torch.float32)
        )
        self.v_uncertainty_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 16, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(16, config.hidden_size, dtype=torch.float32)
        )
        for gate in [self.q_uncertainty_gate, self.k_uncertainty_gate, self.v_uncertainty_gate]:
            torch.nn.init.zeros_(gate[2].weight)
            torch.nn.init.zeros_(gate[2].bias)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Pass 1: Scouting Attention
        query_states_draft = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_draft = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states_draft.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
        cos, sin = self.rotary_emb(key_states_draft, seq_len=kv_seq_len)
        query_states_draft, key_states_draft = apply_rotary_pos_emb(query_states_draft, key_states_draft, cos, sin, position_ids)
        key_states_draft = repeat_kv(key_states_draft, self.num_key_value_groups)
        attn_weights_draft = torch.matmul(query_states_draft, key_states_draft.transpose(2, 3)) / (self.head_dim**0.5)
        if attention_mask is not None:
            attn_weights_draft = attn_weights_draft + attention_mask
        attn_weights_draft = nn.functional.softmax(attn_weights_draft, dim=-1, dtype=torch.float32)

        # Uncertainty Gathering
        with torch.no_grad():
            all_logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
            all_probs = torch.softmax(all_logits.to(torch.float32), dim=-1)
            current_entropy = -torch.sum(all_probs * torch.log(all_probs + 1e-9), dim=-1)
            if past_key_value is not None and kv_seq_len > q_len:
                gathered_uncertainty = current_entropy.unsqueeze(-1)
            else:
                avg_attn_weights_draft = attn_weights_draft.mean(dim=1)
                gathered_uncertainty = torch.matmul(avg_attn_weights_draft, current_entropy.unsqueeze(-1))
            gathered_uncertainty = gathered_uncertainty.to(hidden_states.dtype)

        # Pass 2: Gated Attention
        gate_input = torch.cat([hidden_states.to(torch.float32), gathered_uncertainty.to(torch.float32)], dim=-1)
        q_gate_adj = self.gate_scale * torch.tanh(self.q_uncertainty_gate(gate_input))
        k_gate_adj = self.gate_scale * torch.tanh(self.k_uncertainty_gate(gate_input))
        v_gate_adj = self.gate_scale * torch.tanh(self.v_uncertainty_gate(gate_input))

        if self.analyze_weights:
            # Calculate the average L2 norm of the adjustment vectors for Q, K, V
            # This gives a single "magnitude" value for the gate's effect per token.
            avg_magnitude = (torch.norm(q_gate_adj, p=2, dim=-1) +
                             torch.norm(k_gate_adj, p=2, dim=-1) +
                             torch.norm(v_gate_adj, p=2, dim=-1)) / 3.0
            self.gate_magnitudes.append(avg_magnitude.detach().cpu())

        q_gated_h = hidden_states * (1.0 + q_gate_adj.to(hidden_states.dtype))
        k_gated_h = hidden_states * (1.0 + k_gate_adj.to(hidden_states.dtype))
        v_gated_h = hidden_states * (1.0 + v_gate_adj.to(hidden_states.dtype))

        query_states = self.q_proj(q_gated_h).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(k_gated_h).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(v_gated_h).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim**0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None if not output_attentions else attn_weights, past_key_value

# ==============================================================================
# Model Loading and Patching Functions
# ==============================================================================
def replace_attention_layers(model, lm_head):
    """Replaces standard Qwen2Attention with TwoPassAttention."""
    for i, layer in enumerate(model.model.layers):
        new_attention = TwoPassAttention(model.config, lm_head, layer_idx=i)
        new_attention.load_state_dict(layer.self_attn.state_dict(), strict=False)
        layer.self_attn = new_attention
    return model

def load_model_for_inference(checkpoint_path, base_model_name="Qwen/Qwen2-1.5B-Instruct"):
    """
    Loads a model with a custom architecture from a checkpoint.
    This function first loads the base model, patches it with the custom TwoPassAttention
    architecture, and then loads the fine-tuned weights from the checkpoint.
    """
    print(f"Loading tokenizer from base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print(f"Loading base model architecture from: {base_model_name}")
    # 1. Load the base model architecture and its pretrained weights
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map={'': 'cpu'}
    )
    
    # Add a padding token if it doesn't exist (important for tokenizer consistency)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    print("Patching model with TwoPassAttention architecture...")
    # 2. Replace the standard attention layers with our custom ones.
    # The gate weights are currently zero-initialized.
    model = replace_attention_layers(model, model.lm_head)
    
    print(f"Loading fine-tuned weights from checkpoint: {checkpoint_path}")
    # 3. Load the state dictionary from the checkpoint file(s).
    safetensor_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    if not safetensor_files:
        raise OSError(f"No .safetensors files found in {checkpoint_path}")
        
    checkpoint_state_dict = {}
    for f in sorted(safetensor_files): # Sort to ensure consistent loading order
        checkpoint_state_dict.update(load_file(f, device="cpu"))
        
    # 4. Load the checkpoint weights into our patched model.
    # `strict=False` allows us to load a partial state dict. The checkpoint contains
    # only the trained weights, which will correctly overwrite the base weights
    # and fill in our custom gate weights.
    incompatible_keys = model.load_state_dict(checkpoint_state_dict, strict=False)
    
    # Report any keys that didn't match, which can be useful for debugging.
    if incompatible_keys.missing_keys:
        print(f"Warning: Missing keys in state_dict: {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict: {incompatible_keys.unexpected_keys}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    print("Model loaded and ready for inference.")
    return model, tokenizer

def generate_answer(model, tokenizer, question, max_new_tokens=2048, analyze_weights_influence=False):
    """Generates an answer for a given question."""
    # 1. Reset and prepare the model for analysis if requested
    if analyze_weights_influence:
        for layer in model.model.layers:
            if isinstance(layer.self_attn, TwoPassAttention):
                layer.self_attn.gate_magnitudes = []
                layer.self_attn.analyze_weights = True

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\nGenerating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    # 2. Collect, process, and display analysis results
    if analyze_weights_influence:
        print("\n" + "="*80)
        print("Uncertainty Gate Influence Analysis")
        print("="*80)
        
        all_magnitudes = []
        for i, layer in enumerate(model.model.layers):
            if isinstance(layer.self_attn, TwoPassAttention) and layer.self_attn.gate_magnitudes:
                # Concatenate magnitudes from each generation step. Each step adds a tensor of shape (bsz, 1).
                layer_magnitudes = torch.cat(layer.self_attn.gate_magnitudes, dim=1) # -> (bsz, num_tokens)
                all_magnitudes.append(layer_magnitudes)
                # Reset the analysis flag after use
                layer.self_attn.analyze_weights = False
        
        if all_magnitudes:
            # Stack across layers to get a single tensor: (num_layers, bsz, num_tokens)
            full_magnitudes_tensor = torch.stack(all_magnitudes, dim=0)
            num_generated_tokens = full_magnitudes_tensor.shape[2]

            # Average across layers to get per-token influence: (bsz, num_tokens)
            avg_mag_per_token = full_magnitudes_tensor.mean(dim=0).squeeze(0) # Squeeze bsz if it's 1
            
            # Average across tokens and batch to get per-layer influence: (num_layers)
            avg_mag_per_layer = full_magnitudes_tensor.mean(dim=[1, 2])
            
            print(f"Analysis based on {num_generated_tokens} generated tokens.")
            print("\n>> Average Gate Magnitude per Layer (how much each layer contributed overall):")
            for i, mag in enumerate(avg_mag_per_layer):
                print(f"  Layer {i:2d}: {mag:.6f}")
            
            print(f"\n>> Overall Average Gate Magnitude across all layers and tokens: {full_magnitudes_tensor.mean():.6f}")

            # Find token with maximum gate activation
            if num_generated_tokens > 0:
                max_influence_token_idx = torch.argmax(avg_mag_per_token)
                max_influence_value = avg_mag_per_token[max_influence_token_idx]
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
                token_with_max_influence = generated_tokens[max_influence_token_idx]
                print(f"\n>> Token with maximum gate influence: '{token_with_max_influence}' (at position {max_influence_token_idx})")
                print(f"   - Average Magnitude: {max_influence_value:.6f}")
        else:
            print("No gate magnitude data was recorded. Ensure model is patched correctly.")
        print("="*80)

    return response

# ==============================================================================
# Main Inference Script
# ==============================================================================
def main():
    # --- Hardcoded for notebook execution ---
    # Use the path to your fine-tuned model checkpoint
    checkpoint_path = "./Qwen2-1.5b-two-pass-sft/checkpoint-200/"
    base_model_name = "Qwen/Qwen2-1.5B-Instruct"  # Base model for tokenizer

    # Set a prompt here to run on a single question, or leave as None for interactive mode
    # Example: prompt = "What is the capital of France?"
    prompt = None

    # Set to True to analyze the influence of the uncertainty gates
    analyze_weights_influence = True
    # --- End of hardcoded section ---

    print(f"Using checkpoint: {checkpoint_path}")
    model, tokenizer = load_model_for_inference(checkpoint_path, base_model_name=base_model_name)

    if prompt:
        response = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=2048,
            analyze_weights_influence=analyze_weights_influence
        )
        print("\n" + "="*80)
        print("Model Response:")
        print("="*80)
        print(response)
    else:
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print("Enter your prompt below. Type 'exit' or 'quit' to end.")
        while True:
            try:
                question = input("\nPrompt: ")
                if question.lower() in ["exit", "quit"]:
                    break
                response = generate_answer(
                    model,
                    tokenizer,
                    question,
                    max_new_tokens=2048,
                    analyze_weights_influence=analyze_weights_influence
                )
                print("\n" + "-"*50)
                print("Response:")
                print(response)
                print("-"*50)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting interactive mode.")

if __name__ == "__main__":
    main()
