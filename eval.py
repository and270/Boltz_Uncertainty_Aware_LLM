import pandas as pd
import re
from tqdm import tqdm
from safetensors.torch import load_file
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime
import torch.nn as nn
from datasets import load_dataset
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv

# ==============================================================================
# CONFIG
# ==============================================================================
CONFIG = {
    'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
    'output_dir': './Qwen2.5-1.5b-two-pass-sft',
    'max_seq_length': 2048,
    'EVALUATE_BASE_MODEL': True,  # Set to False to only evaluate the trained model
    'PRINT_INTERVAL': 5,         # Print progress every N samples
    'SHOW_RESPONSES': True,      # Set to True to show question and responses during progress
}

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
# Helper Functions
# ==============================================================================
def replace_attention_layers(model, lm_head):
    """Replaces standard attention layers with TwoPassAttention."""
    for i, layer in enumerate(model.model.layers):
        new_attention = TwoPassAttention(model.config, lm_head, layer_idx=i)
        layer.self_attn = new_attention
    return model

print("\n" + "="*80)
print("GSM8K EVALUATION: TWO-PASS ATTENTION MODEL")
print("="*80)

def extract_last_number(text):
    numbers = re.findall(r'-?\$?[\d,]*\.?\d+', text)
    if not numbers:
        return None
    return float(numbers[-1].replace('$', '').replace(',', ''))

def load_and_prepare_models_for_eval():
    print("\nLoading models for evaluation...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    original_model, original_tokenizer = None, None
    if CONFIG['EVALUATE_BASE_MODEL']:
        # 1. Load original model
        print("  Loading original Qwen model...")
        original_tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
        original_model = AutoModelForCausalLM.from_pretrained(CONFIG['model_name'], torch_dtype=torch.float32).to(device)
        if original_tokenizer.pad_token is None:
            original_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            original_model.resize_token_embeddings(len(original_tokenizer))
        original_model.eval()

    # 2. Load the custom Two-Pass model
    print(f"  Loading Two-Pass model from {CONFIG['output_dir']}...")
    two_pass_tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    base_model_for_eval = AutoModelForCausalLM.from_pretrained(CONFIG['model_name'], torch_dtype=torch.float32, device_map={'':'cpu'})
    two_pass_model = replace_attention_layers(base_model_for_eval, base_model_for_eval.lm_head)
    
    safetensor_files = glob.glob(os.path.join(CONFIG['output_dir'], "*.safetensors"))
    if not safetensor_files:
        raise OSError(f"No .safetensors files found in {CONFIG['output_dir']}")
        
    final_model_state_dict = {}
    for f in sorted(safetensor_files):
        final_model_state_dict.update(load_file(f, device="cpu"))
        
    two_pass_model.load_state_dict(final_model_state_dict, strict=False)
    two_pass_model = two_pass_model.to(device)
    two_pass_model.eval()

    print("  All models loaded and in eval mode.")
    return (original_model, original_tokenizer), (two_pass_model, two_pass_tokenizer)

def generate_answer(model, tokenizer, question, max_new_tokens=2048):
    messages = [
        {"role": "user", "content": f"Please solve the following math problem. Your answer must end with the final numerical result in the format: ####<final answer>. \n\nProblem: {question}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=0.6,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

# --- Main Evaluation Logic ---
(original_model, original_tokenizer), (two_pass_model, two_pass_tokenizer) = load_and_prepare_models_for_eval()
gsm8k_dataset = load_dataset("gsm8k", "main")
test_samples = gsm8k_dataset["test"].select(range(300)) # Evaluate on 300 samples

results = []
original_correct, two_pass_correct = 0, 0

for i, example in enumerate(tqdm(test_samples, desc="Evaluating GSM8K")):
    question = example["question"]
    true_answer = extract_last_number(example["answer"])
    
    # Evaluate Two-Pass Model
    two_pass_answer_text = generate_answer(two_pass_model, two_pass_tokenizer, question)
    two_pass_answer = extract_last_number(two_pass_answer_text)
    two_pass_is_correct = abs(two_pass_answer - true_answer) < 0.01 if two_pass_answer is not None and true_answer is not None else False
    if two_pass_is_correct:
        two_pass_correct += 1

    result_data = {
        'question': question,
        'true_answer': true_answer,
        'two_pass_answer': two_pass_answer,
        'two_pass_correct': two_pass_is_correct
    }

    # Evaluate Original Model (if enabled)
    if CONFIG['EVALUATE_BASE_MODEL'] and original_model:
        original_answer_text = generate_answer(original_model, original_tokenizer, question)
        original_answer = extract_last_number(original_answer_text)
        original_is_correct = abs(original_answer - true_answer) < 0.01 if original_answer is not None and true_answer is not None else False
        if original_is_correct:
            original_correct += 1
        result_data['original_answer'] = original_answer
        result_data['original_correct'] = original_is_correct

    results.append(result_data)
    
    # Print live progress
    if (i + 1) % CONFIG['PRINT_INTERVAL'] == 0:
        current_total = i + 1
        live_two_pass_accuracy = two_pass_correct / current_total
        
        progress_msg = f"\n--- Progress at sample {current_total}/{len(test_samples)} ---\n"
        progress_msg += f"  Two-Pass Model Accuracy: {live_two_pass_accuracy:.2%} ({two_pass_correct}/{current_total})\n"
        
        if CONFIG['EVALUATE_BASE_MODEL']:
            live_original_accuracy = original_correct / current_total
            progress_msg += f"  Original Model Accuracy: {live_original_accuracy:.2%} ({original_correct}/{current_total})\n"
        
        # Show current question and responses if enabled
        if CONFIG['SHOW_RESPONSES']:
            progress_msg += f"\n  Current Question: {question}\n"
            progress_msg += f"  True Answer: {true_answer}\n"
            progress_msg += f"  Two-Pass Response: {two_pass_answer_text}\n"
            progress_msg += f"  Two-Pass Answer: {two_pass_answer} {'✓' if two_pass_is_correct else '✗'}\n"
            
            if CONFIG['EVALUATE_BASE_MODEL'] and 'original_answer_text' in locals():
                progress_msg += f"  Original Response: {original_answer_text}\n"
                progress_msg += f"  Original Answer: {original_answer} {'✓' if original_is_correct else '✗'}\n"
            
        tqdm.write(progress_msg.strip())


# --- Display Results ---
total = len(results)
two_pass_accuracy = two_pass_correct / total if total > 0 else 0

print("\n" + "="*80)
print("FINAL EVALUATION RESULTS")
print("="*80)
print(f"Total samples evaluated: {total}")

if CONFIG['EVALUATE_BASE_MODEL']:
    original_accuracy = original_correct / total if total > 0 else 0
    print(f"\nORIGINAL MODEL:")
    print(f"  Correct answers: {original_correct}")
    print(f"  Accuracy: {original_accuracy:.2%}")

print(f"\nTWO-PASS ATTENTION MODEL:")
print(f"  Correct answers: {two_pass_correct}")
print(f"  Accuracy: {two_pass_accuracy:.2%}")

if CONFIG['EVALUATE_BASE_MODEL']:
    print(f"\nPERFORMANCE CHANGE: {two_pass_accuracy - original_accuracy:+.2%}")
    if two_pass_accuracy > original_accuracy:
        print("\n🎉 The Two-Pass Attention model performs BETTER!")
    elif two_pass_accuracy == original_accuracy:
        print("\n- Both models have similar performance.")
    else:
        print("\n📉 The original fine-tuned model performs better.")


df = pd.DataFrame(results)
print("\nSample of results:")
print(df.head().to_string())

# Save results to csv
results_filename = f"two_pass_evaluation_results_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
df.to_csv(results_filename, index=False)
print(f"\nFull results saved to {results_filename}")