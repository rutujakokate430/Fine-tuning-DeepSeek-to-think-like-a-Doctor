# Fine-Tuning DeepSeek-R1-Distill-Llama-8B with LoRA to think like a Doctor

## Overview

This project fine-tunes the **DeepSeek-R1-Distill-Llama-8B** model using **LoRA (Low-Rank Adaptation)** technique to efficiently adapt a large-scale LLM with reduced memory and compute requirements.

We leverage **Unsloth’s FastLanguageModel** to optimize the loading and training of the model while using **4-bit quantization** to minimize VRAM consumption. The fine-tuning process is tracked and visualized using **Weights & Biases (W&B)**.

## Why LoRA?

### LoRA (Low-Rank Adaptation)

LoRA optimizes model adaptation by **freezing the original model** and adding trainable **low-rank matrices (A, B)** to specific layers:

- **Memory Efficient**: Only a small number of parameters are trained.
- **Fast Adaptation**: Avoids full model retraining.
- **Improved Performance**: Fine-tunes only the task-relevant subspace.

Mathematically, instead of updating the full weight matrix \(W\), we optimize:
\[ W + \Delta W, \quad \text{where} \quad \Delta W = A \times B \]
where \(A, B\) are low-rank matrices.

<img width="481" alt="image" src="https://github.com/user-attachments/assets/797a4d39-81f0-4006-bb66-2b0e3b31e309" />

---

## Model & Dataset

### Model: `DeepSeek-R1-Distill-Llama-8B`

- **8B parameters** (distilled from a larger DeepSeek model)
- **4-bit quantization enabled**
- **Max sequence length: 2048 tokens**

### Dataset:

- **Source:** `BothBosu/scam-dialogue`
- **Size:** 600 training samples (subset for experimentation)
- **Structure:** Includes questions, chain-of-thought reasoning, and responses

---
## Tech Stack

| Component              | Technology Used                                  | Purpose                                        |
|----------------------|------------------------------------------------|------------------------------------------------|
| **Model**            | DeepSeek-R1-Distill-Llama-8B                   | Base large language model for fine-tuning     |
| **Fine-Tuning**      | LoRA, QLoRA                                    | Efficient model adaptation with fewer parameters |
| **Framework**        | PyTorch, Transformers (Hugging Face)           | Deep learning framework for model training    |
| **Model Optimization** | FastLanguageModel (Unsloth)                    | Optimized model loading and execution        |
| **Data Processing**   | Hugging Face Datasets, Pandas                  | Handling and preprocessing training data     |
| **Training Logging**  | Weights & Biases (W&B)                         | Tracking training metrics and performance     |
| **Quantization**      | 4-bit Quantization, Bitsandbytes                | Reducing VRAM usage while maintaining performance |
| **Hardware**         | NVIDIA RTX 3090/4090, A100 GPUs                 | Compute resources for fine-tuning             |

## Fine-Tuning Configuration

### LoRA Parameters

| Parameter         | Value | Explanation                                                                               |
| ----------------- | ----- | ----------------------------------------------------------------------------------------- |
| **LoRA Rank (r)** | 16    | Number of trainable low-rank matrices. Higher = better adaptation, but more memory usage. |
| **LoRA Alpha**    | 16    | Scaling factor for LoRA weights.                                                          |
| **LoRA Dropout**  | 0     | No dropout to retain full training signal.                                                |
| **Bias**          | None  | No additional bias terms are trained.                                                     |

### Training Hyperparameters

| Parameter                 | Value        | Explanation                                         |
| ------------------------- | ------------ | --------------------------------------------------- |
| **Batch Size**            | 2            | Small batch size to fit within memory.              |
| **Gradient Accumulation** | 4            | Simulates larger batch size without exceeding VRAM. |
| **Epochs**                | 1            | Single epoch to prevent overfitting.                |
| **Max Steps**             | 60           | Limits training steps for quick experimentation.    |
| **Learning Rate**         | 2e-4         | Adjusted for stable convergence.                    |
| **Warmup Steps**          | 5            | Gradually increases LR for stability.               |
| **Weight Decay**          | 0.01         | Prevents overfitting.                               |
| **Optimizer**             | `adamw_8bit` | Efficient memory usage.                             |

### Calculation of Training Steps

Total steps required for one epoch:
\[
\text{Total Steps} = \frac{\text{Dataset Size}}{\text{Batch Size} \times \text{Number of GPUs} \times \text{Gradient Accumulation Steps}} \times \text{Epochs}
\]
\[
= \frac{600}{2 \times 2 \times 4} \times 1 = 37 \text{ steps (rounded)}
\]

Since `max_steps=60`, we allow for additional training beyond one epoch.

### Memory Optimization

| Feature                    | Enabled? | Benefit                            |
| -------------------------- | -------- | ---------------------------------- |
| **4-bit Quantization**     | ✅        | Reduces memory usage by \~75%.     |
| **BF16 / FP16 Precision**  | ✅        | Optimized for modern GPUs.         |
| **Gradient Checkpointing** | ✅        | Saves VRAM during backpropagation. |

---
<img width="385" alt="image" src="https://github.com/user-attachments/assets/7da9856d-43af-45ef-9bac-4f87096ab99e" />

## Challenges Faced

- **Memory Constraints:** Even with 4-bit quantization, fitting an 8B model required careful tuning of batch size and gradient accumulation.
- **Stability of LoRA Rank:** Experimenting with `r=8` led to underfitting, while `r=32` consumed excessive memory.
- **Gradient Exploding:** Initial training showed unstable gradients, requiring `gradient_accumulation_steps=4` to stabilize updates.

## Training Curves
<img width="516" alt="image" src="https://github.com/user-attachments/assets/ccac3d48-2e95-4d34-b782-0c9289865a5a" />




