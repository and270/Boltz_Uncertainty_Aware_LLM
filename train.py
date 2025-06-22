# ==============================================================================
# CELL 1: MARKDOWN
# ==============================================================================
# ## Uncertainty-Aware Transformers: Two-Pass Attention Implementation
#
# This notebook implements the full solution for **Option B: The Relationship-Based, Two-Pass Attention Approach**.
#
# ### Core Concepts:
# 1.  **`TwoPassAttention` Module**: A new attention mechanism that replaces the standard `Qwen2Attention`.
# 2.  **Scouting Pass (Pass 1)**: Computes draft attention scores (`Q_draft @ K_draft.T`) to determine token relationships without applying any gates.
# 3.  **Uncertainty Gathering**: Calculates Softmax Entropy for all tokens and uses the draft attention scores to create a relevance-weighted uncertainty signal for each token.
# 4.  **Gating Pass (Pass 2)**: Uses this powerful, non-local uncertainty signal to gate the Q, K, and V inputs before the final attention calculation, allowing the model to focus its reasoning.
# 5.  **Causality**: This two-pass design elegantly solves the circular dependency problem inherent in using attention scores to inform the same attention calculation.

# ==============================================================================
# CELL 2: Package Installation
# ==============================================================================
# 1. Install the correct version of PyTorch for your CUDA 11.8 environment.
# !pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 2. Install the pinned, compatible versions of the Hugging Face ecosystem libraries.
# !pip install transformers==4.41.2 accelerate==0.29.3 datasets==2.18.0 tokenizers==0.19.1

# 3. Install other necessary utilities including wandb for experiment tracking.
# !pip install numpy pandas tqdm wandb

print("All packages installed successfully with correct compatible versions!")

# ==============================================================================
# CELL 3: Setup and Imports
# ==============================================================================
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
import transformers
import wandb
import os
from datetime import datetime

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# CELL 4: Configuration Parameters
# ==============================================================================
# Training configuration for the Two-Pass Attention model
CONFIG = {
    'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
    'dataset_name': 'open-r1/OpenR1-Math-220k',
    'output_dir': './Qwen2.5-1.5b-two-pass-sft',
    'training_epochs': 1,
    'batch_size': 1,  # Smaller batch size due to increased memory usage of two-pass attention
    'gradient_accumulation_steps': 8, # Increase accumulation to maintain effective batch size
    'learning_rate': 1e-5,
    'max_seq_length': 2048,
    'wandb_project': 'uncertainty-aware-transformers',
    'wandb_name': f'Qwen2.5-1.5b-two-pass-sft-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
}

print("Training Configuration for Two-Pass Model:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# Initialize wandb
print("\nInitializing Weights & Biases...")
wandb.init(
    project=CONFIG['wandb_project'],
    name=CONFIG['wandb_name'],
    config=CONFIG,
    tags=['qwen2', 'two-pass-attention', 'softmax-entropy', 'full-model-sft']
)

# ==============================================================================
# CELL 5: Two-Pass Attention Implementation
# ==============================================================================
class TwoPassAttention(Qwen2Attention):
    """
    Implements a two-pass attention mechanism where the first pass scouts for token relationships
    and the second pass uses this information to gate the Q, K, and V projections.
    """
    def __init__(self, config, lm_head, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)

        # Store a reference to the language model head to calculate entropy
        self.lm_head = lm_head

        # Gate input dimension: hidden_size (from token) + 1 (from gathered uncertainty)
        gate_input_dim = config.hidden_size + 1

        # Scaling factor for the additive residual gate
        self.gate_scale = 0.2

        # Separate gate controllers for Q, K, and V
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

        # Initialize final layer of gates to zero for identity initialization
        for gate in [self.q_uncertainty_gate, self.k_uncertainty_gate, self.v_uncertainty_gate]:
            torch.nn.init.zeros_(gate[2].weight)
            torch.nn.init.zeros_(gate[2].bias)

    def forward(
        self,
        hidden_states, # shape: (bsz, q_len, dim)
        attention_mask=None, # shape: (bsz, 1, q_len, kv_len)
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # === PASS 1: SCOUTING ATTENTION ===
        # Use ungated hidden states to find token relationships.
        # This section mirrors the standard attention logic to get draft weights.

        # Projections and reshaping for draft Q and K
        query_states_draft = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_draft = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states_draft.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        cos, sin = self.rotary_emb(key_states_draft, seq_len=kv_seq_len)
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
        query_states_draft, key_states_draft = apply_rotary_pos_emb(query_states_draft, key_states_draft, cos, sin, position_ids)

        if past_key_value is not None:
             # In a real generation scenario, the logic here would need to manage a draft cache.
             # For simplicity in training, we proceed without draft caching.
             pass

        key_states_draft = repeat_kv(key_states_draft, self.num_key_value_groups)

        attn_weights_draft = torch.matmul(query_states_draft, key_states_draft.transpose(2, 3)) / (self.head_dim**0.5)
        if attention_mask is not None:
            attn_weights_draft = attn_weights_draft + attention_mask
        attn_weights_draft = nn.functional.softmax(attn_weights_draft, dim=-1, dtype=torch.float32)


        # === UNCERTAINTY GATHERING ===
        # Use the draft attention weights to gather uncertainty from relevant tokens.
        # This entire block does not participate in backpropagation to keep the gradient path clean.
        with torch.no_grad():
            # 1. Calculate softmax entropy for current query tokens only
            all_logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
            all_probs = torch.softmax(all_logits.to(torch.float32), dim=-1)
            current_entropy = -torch.sum(all_probs * torch.log(all_probs + 1e-9), dim=-1) # Shape: (bsz, q_len)

            # 2. For generation with past_key_value, we need to handle uncertainty differently
            if past_key_value is not None and kv_seq_len > q_len:
                # During generation, we only have current token's entropy
                # Create a simple uncertainty signal based on current entropy
                gathered_uncertainty = current_entropy.unsqueeze(-1)  # Shape: (bsz, q_len, 1)
            else:
                # During training, we have full sequence
                # Use draft attention weights to create a relevance-weighted uncertainty signal
                # Average attention heads to get a single relationship matrix
                avg_attn_weights_draft = attn_weights_draft.mean(dim=1) # Shape: (bsz, q_len, kv_len)
                
                # For training, kv_len == q_len, so we can use matrix multiplication
                gathered_uncertainty = torch.matmul(avg_attn_weights_draft, current_entropy.unsqueeze(-1)) # Shape: (bsz, q_len, 1)
            
            gathered_uncertainty = gathered_uncertainty.to(hidden_states.dtype)

        # === PASS 2: GATED ATTENTION ===
        # Use the gathered uncertainty to modulate the final Q, K, V projections.

        # 1. Prepare input for the gate controllers
        gate_input = torch.cat([hidden_states.to(torch.float32), gathered_uncertainty.to(torch.float32)], dim=-1)

        # 2. Compute gate adjustments using the Additive Residual Gate formulation
        q_gate_adj = self.gate_scale * torch.tanh(self.q_uncertainty_gate(gate_input))
        k_gate_adj = self.gate_scale * torch.tanh(self.k_uncertainty_gate(gate_input))
        v_gate_adj = self.gate_scale * torch.tanh(self.v_uncertainty_gate(gate_input))

        # 3. Apply gates to hidden states: gated_h = h * (1 + adjustment)
        q_gated_h = hidden_states * (1.0 + q_gate_adj.to(hidden_states.dtype))
        k_gated_h = hidden_states * (1.0 + k_gate_adj.to(hidden_states.dtype))
        v_gated_h = hidden_states * (1.0 + v_gate_adj.to(hidden_states.dtype))

        # 4. Perform the final, gated attention calculation (standard logic from here)
        query_states = self.q_proj(q_gated_h).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(k_gated_h).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(v_gated_h).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to the final Q, K
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim**0.5)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(f"Attention weights have incorrect size: {attn_weights.size()}")
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

print("TwoPassAttention class defined successfully!")

# ==============================================================================
# CELL 6: Attention Layer Replacement Function (Modified)
# ==============================================================================
def replace_attention_layers(model, lm_head):
    """Replaces standard attention layers with TwoPassAttention.

    Args:
        model: The model to modify.
        lm_head: A reference to the model's language model head, required by TwoPassAttention.
    """
    for i, layer in enumerate(model.model.layers):
        print(f"Replacing attention in layer {i}...")

        # Create the new attention module, passing the lm_head reference
        new_attention = TwoPassAttention(model.config, lm_head, layer_idx=i)

        # Copy weights from the original attention layer
        new_attention.load_state_dict(layer.self_attn.state_dict(), strict=False)

        # Replace the old module
        layer.self_attn = new_attention

    print("\nAll attention layers replaced with TwoPassAttention.")
    return model

print("replace_attention_layers function defined to handle lm_head passing!")

# ==============================================================================
# CELL 7: Load Base Model and Tokenizer
# ==============================================================================
print(f"Loading base model and tokenizer: {CONFIG['model_name']}")

tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

# Load model in float32 for stability with custom layers
model = AutoModelForCausalLM.from_pretrained(
    CONFIG['model_name'],
    torch_dtype=torch.float32,
    device_map={'': 'cpu'} # Load on CPU first before modifying and moving to GPU
)
print("Model loaded successfully on CPU with default attention!")

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Added padding token and updated model config.")

# ==============================================================================
# CELL 8: Replace Attention and Finalize Model Setup
# ==============================================================================
# 1. Replace attention layers. This must be done BEFORE moving model to GPU.
print("Building Two-Pass architecture...")
model = replace_attention_layers(model, model.lm_head) # Pass the lm_head

# 2. Move the entire modified model to the GPU
print("\nMoving modified model to cuda:0...")
model = model.to('cuda:0')
print("Model is now on GPU.")

# 3. Count parameters
gate_params = sum(p.numel() for name, p in model.named_parameters() if 'uncertainty_gate' in name)
total_params = sum(p.numel() for p in model.parameters())

print(f"\nModel statistics after replacement:")
print(f"  Total parameters: {total_params:,}")
print(f"  Uncertainty gate parameters: {gate_params:,}")
print(f"  Base model parameters: {total_params - gate_params:,}")
print(f"  Gate parameters ratio: {gate_params/total_params:.4%}")

# 4. Final verification
devices = {p.device for p in model.parameters()}
print(f"Final model device(s): {devices}")
assert len(devices) == 1 and next(iter(devices)) == torch.device('cuda:0'), "Model is not fully on GPU!"

# ==============================================================================
# CELL 9: Load and Preprocess Dataset
# ==============================================================================
print(f"\nLoading dataset: {CONFIG['dataset_name']}")
dataset = load_dataset(CONFIG['dataset_name'])
#dataset['train'] = dataset['train'].select(range(2000))

def tokenize_function(examples):
    """
    Tokenize the dataset using the model's chat template.
    """
    formatted_inputs = []
    for i in range(len(examples["problem"])):
        generations = examples["generations"][i]
        correctness = examples["correctness_math_verify"][i]

        selected_generation = generations[0] # Default to first generation
        for gen, is_correct in zip(generations, correctness):
            if is_correct:
                selected_generation = gen
                break

        conversation = [
            {"role": "user", "content": examples["problem"][i]},
            {"role": "assistant", "content": selected_generation}
        ]

        formatted_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        formatted_inputs.append(formatted_text)

    tokenized = tokenizer(
        formatted_inputs,
        padding=False,
        truncation=False
    )
    return tokenized

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing all examples"
)

# Filter out examples that are too long
train_dataset = tokenized_datasets["train"].filter(
    lambda example: len(example['input_ids']) <= CONFIG['max_seq_length'],
    desc="Filtering by length"
)

print(f"Dataset size after filtering: {len(train_dataset)}")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print("Data collator for Causal LM initialized.")

# ==============================================================================
# CELL 10: Model Training
# ==============================================================================
print("\n" + "="*60)
print("TWO-PASS ATTENTION MODEL TRAINING")
print("="*60)

# Unfreeze all parameters for full model fine-tuning
for name, param in model.named_parameters():
    param.requires_grad = True

training_args = TrainingArguments(
    output_dir=CONFIG['output_dir'],
    num_train_epochs=CONFIG['training_epochs'],
    per_device_train_batch_size=CONFIG['batch_size'],
    gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
    learning_rate=CONFIG['learning_rate'],
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=f"{CONFIG['output_dir']}/logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    report_to="wandb",
    run_name=CONFIG['wandb_name'],
    bf16=False, # Keep in float32 for stability
    fp16=False,
    ddp_find_unused_parameters=False, # Important for custom architectures
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
)

# Custom trainer to log gate gradients
class TwoPassTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: dict) -> torch.Tensor:
        loss = super().training_step(model, inputs)

        # Log the norm of the gradients for the uncertainty gates
        gate_grad_norms = {}
        for name, param in model.named_parameters():
            if "uncertainty_gate" in name and param.grad is not None:
                gate_grad_norms[f"grads_norm/{name}"] = param.grad.data.norm(2).item()
        if gate_grad_norms:
            self.log(gate_grad_norms)

        return loss

trainer = TwoPassTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

print("\nTrainer initialized for Two-Pass Attention model.")
print(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")

# Start training
trainer.train()

print("\nTraining completed!")

# Save final model
print(f"Saving final model to: {CONFIG['output_dir']}")
model.save_pretrained(CONFIG['output_dir'])
tokenizer.save_pretrained(CONFIG['output_dir'])
print("Model saved successfully.")

wandb.finish()


