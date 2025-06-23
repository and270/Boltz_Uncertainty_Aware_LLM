# Uncertainty-Aware Transformers via Two-Pass Attention

This repository contains the implementation of an experimental Transformer architecture designed to make the attention mechanism aware of the model's internal uncertainty. The core idea is a **Two-Pass Attention** mechanism that uses a "scouting pass" to inform a "gating pass".

## The Core Idea

Large Language Models often struggle with quantifying their own uncertainty, sometimes producing confident but incorrect answers. This project explores an architectural modification to address this. The hypothesis is that if the model's attention mechanism can be made aware of token-level uncertainty (i.e., how "confused" the model is about a particular token's role or a subsequent prediction), it can learn to allocate its representational capacity more effectively.

For instance, in a complex reasoning task, the model could learn to pay more attention to parts of the context that resolve its uncertainty, or alternatively, to suppress noisy or ambiguous signals.

## The Method: Two-Pass Attention

To solve the causality problem of needing attention weights to calculate an uncertainty signal that in turn influences those same weights, we introduce a two-pass mechanism within each attention block. This replaces the standard self-attention layer.

1.  **Pass 1: Scouting Pass**
    *   A standard, ungated attention calculation is performed (`Q_draft @ K_draft.T`).
    *   This produces a set of `attn_weights_draft`. These weights represent the initial, unfiltered relationships between tokens in the sequence.
    *   This pass does **not** modify the hidden states and is used only for information gathering.

2.  **Uncertainty Gathering**
    *   With the draft attention weights, we gather an uncertainty signal. We use **Softmax Entropy** as our uncertainty metric.
    *   For each token, we calculate its predictive entropy using the model's language model head (`lm_head`).
    *   The draft attention weights are then used to compute a relevance-weighted average of these entropies for each token. This creates a `gathered_uncertainty` vector, where each token's uncertainty signal is informed by the uncertainty of the tokens it's related to.
    *   This entire step is performed within a `torch.no_grad()` context to prevent it from affecting the main gradient path of the scouting pass, keeping the training signal clean.

3.  **Pass 2: Gated Attention**
    *   The `gathered_uncertainty` signal is concatenated with the original `hidden_states`.
    *   This combined tensor is fed into three small, separate neural networks (the "gates") to compute an adjustment for the Query (`Q`), Key (`K`), and Value (`V`) projections.
    *   The adjustment is applied additively: `gated_hidden_state = hidden_state * (1 + gate_adjustment)`. This allows the model to learn to amplify or dampen the inputs to the final attention calculation based on the uncertainty context.
    *   A final, standard attention calculation is performed using these gated `Q`, `K`, and `V` states to produce the layer's output.

## Code Structure

*   `train.py`: This script handles the full fine-tuning process.
    *   It loads a base model (`Qwen/Qwen2.5-1.5B-Instruct`).
    *   It defines the `TwoPassAttention` class and includes a function to patch the base model's architecture, replacing the standard attention layers.
    *   It prepares the `open-r1/OpenR1-Math-220k` dataset for training.
    *   It uses the Hugging Face `Trainer` to perform supervised fine-tuning.

*   `eval.py`: This script is used to evaluate the performance of the fine-tuned model.
    *   It loads the custom `TwoPassAttention` model from a checkpoint.
    *   For comparison, it also loads the original base model.
    *   It runs both models on the `gsm8k` test set, comparing their accuracy on math word problems.
    *   It saves the results to a `.csv` file for analysis.

*   `inference.py`: A script for interacting with the fine-tuned model.
    *   Loads the model from a checkpoint using the same patching mechanism as `train.py`.
    *   Provides an interactive command-line interface to chat with the model.
    *   Includes a mode to analyze the influence of the uncertainty gate activations during generation, offering a glimpse into the model's internal workings.

## How to Run

1.  **Setup**: Install the required packages. It is recommended to use a virtual environment.
    ```bash
    # Pinned versions from the original project setup
    pip install torch torchvision torchaudio
    pip install transformers accelerate datasets tokenizers
    pip install pandas tqdm wandb safetensors
    ```

2.  **Training**: Run the training script. This will download the base model and dataset, and save checkpoints to the output directory specified in the script (`./Qwen2.5-1.5b-two-pass-sft`).
    ```bash
    python train.py
    ```

3.  **Evaluation**: After training, run the evaluation script to compare the custom model against the base model on GSM8K.
    ```bash
    python eval.py
    ```

4.  **Inference**: To chat with your fine-tuned model, run the inference script. You may need to update the `checkpoint_path` variable inside the script first.
    ```bash
    python inference.py
    ```

## Results and Conclusion

The evaluation results on the GSM8K benchmark have **not** shown a definitive improvement in performance for the Two-Pass Attention model compared to a standard fine-tuned baseline. The added architectural complexity did not translate into higher accuracy on this specific task.

However, this project serves as a valuable exploration into building uncertainty-aware mechanisms directly into a model's architecture. The two-pass design is a viable pattern for avoiding causal paradoxes in attention modifications. This architecture might serve as a general idea for future improvements. Future work could explore different uncertainty metrics, alternative gating mechanisms, or test this architecture on tasks where explicit uncertainty quantification is more critical (e.g., open-ended generation, evidence-based QA, or detecting out-of-distribution inputs).

## License

This project is licensed for **research and non-commercial use only**. The code is provided as-is, without warranty. You may use, distribute, and modify the code for academic and research purposes, but you may not use it for any commercial products or services. 