
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
import transformers
import wandb


print("\n" + "="*80)
print("GSM8K EVALUATION: ORIGINAL vs. TWO-PASS ATTENTION MODEL")
print("="*80)

def extract_last_number(text):
    numbers = re.findall(r'-?\$?[\d,]*\.?\d+', text)
    if not numbers:
        return None
    return float(numbers[-1].replace('$', '').replace(',', ''))

def load_and_prepare_models_for_eval():
    print("\nLoading models for evaluation...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 1. Load original model
    print("  Loading original Qwen model...")
    original_tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    original_model = AutoModelForCausalLM.from_pretrained(CONFIG['model_name'], torch_dtype=torch.float32).to(device)
    if original_tokenizer.pad_token is None:
        original_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        original_model.resize_token_embeddings(len(original_tokenizer))

    # 2. Load the custom Two-Pass model with the corrected procedure
    print(f"  Loading Two-Pass model from {CONFIG['output_dir']}...")
    
    # Load tokenizer from the base model, not the checkpoint
    two_pass_tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    # First, build the correct base architecture on CPU
    base_model_for_eval = AutoModelForCausalLM.from_pretrained(CONFIG['model_name'], torch_dtype=torch.float32, device_map={'':'cpu'})
    
    # Patch the architecture with our custom layers
    two_pass_model = replace_attention_layers(base_model_for_eval, base_model_for_eval.lm_head)
    
    # Find and load the fine-tuned weights from the final output directory
    safetensor_files = glob.glob(os.path.join(CONFIG['output_dir'], "*.safetensors"))
    if not safetensor_files:
        raise OSError(f"No .safetensors files found in {CONFIG['output_dir']}")
        
    final_model_state_dict = {}
    for f in sorted(safetensor_files):
        final_model_state_dict.update(load_file(f, device="cpu"))
        
    # Load the fine-tuned state dict into the patched architecture
    two_pass_model.load_state_dict(final_model_state_dict, strict=False)
    
    # Move the final model to the device
    two_pass_model = two_pass_model.to(device)

    two_pass_model.eval()
    original_model.eval()

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
            pad_token_id=tokenizer.eos_token_id # Use eos_token_id for open-ended generation
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

# --- Main Evaluation Logic ---
(original_model, original_tokenizer), (two_pass_model, two_pass_tokenizer) = load_and_prepare_models_for_eval()
gsm8k_dataset = load_dataset("gsm8k", "main")
test_samples = gsm8k_dataset["test"].select(range(300)) # Evaluate on 300 samples

results = []
original_correct, two_pass_correct = 0, 0

for example in tqdm(test_samples, desc="Evaluating GSM8K"):
    question = example["question"]
    true_answer = extract_last_number(example["answer"])

    original_answer_text = generate_answer(original_model, original_tokenizer, question)
    original_answer = extract_last_number(original_answer_text)

    two_pass_answer_text = generate_answer(two_pass_model, two_pass_tokenizer, question)
    two_pass_answer = extract_last_number(two_pass_answer_text)

    original_is_correct = abs(original_answer - true_answer) < 0.01 if original_answer is not None and true_answer is not None else False
    two_pass_is_correct = abs(two_pass_answer - true_answer) < 0.01 if two_pass_answer is not None and true_answer is not None else False

    if original_is_correct: original_correct += 1
    if two_pass_is_correct: two_pass_correct += 1

    results.append({
        'question': question,
        'true_answer': true_answer,
        'original_answer': original_answer,
        'two_pass_answer': two_pass_answer,
        'original_correct': original_is_correct,
        'two_pass_correct': two_pass_is_correct
    })

# --- Display Results ---
total = len(results)
original_accuracy = original_correct / total if total > 0 else 0
two_pass_accuracy = two_pass_correct / total if total > 0 else 0

print("\n" + "="*80)
print("FINAL EVALUATION RESULTS")
print("="*80)
print(f"Total samples evaluated: {total}")
print(f"\nORIGINAL MODEL:")
print(f"  Correct answers: {original_correct}")
print(f"  Accuracy: {original_accuracy:.2%}")
print(f"\nTWO-PASS ATTENTION MODEL:")
print(f"  Correct answers: {two_pass_correct}")
print(f"  Accuracy: {two_pass_accuracy:.2%}")
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