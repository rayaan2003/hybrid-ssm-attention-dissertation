# Hybrid SSM-Attention Networks: Architectural Mixing Strategies

**MSc Artificial Intelligence Dissertation**
Queen Mary University of London

**Author:** Rayaan Sheikh
**Supervisor:** Fredrik Dahlqvist

---

## Overview

This repository contains the full implementation, experiments, and
results for the dissertation *"Architectural Mixing Strategies for
Hybrid SSM-Attention Networks: A Systematic Study of Efficiency and
Generalisation."*

The project investigates how a Transformer and a Mamba (state space)
model should actually be combined into a hybrid, rather than assuming a
particular design in advance. Two mixing mechanisms are formalised and
tested:

1. **Stochastic routing** — a policy that selects one whole model per
   input. This is shown, both theoretically and empirically, to always
   collapse to whichever pure model is better, since its objective is
   affine in the mixing parameter $\lambda$.
2. **Probabilistic blending** — a mechanism that mixes the two models'
   output distributions directly. This is provably better-behaved
   (convex, via Jensen's inequality) and is shown empirically to reach a
   genuine interior optimum that outperforms both pure baselines.

The project also measures inference latency across a range of sequence
lengths to test the underlying $\mathcal{O}(n)$ vs $\mathcal{O}(n^2)$
complexity claim directly, and uses Centred Kernel Alignment (CKA) to
compare the two models' learned representations.

## Key Results

| Model | Val. Loss | Perplexity | Latency (ms/token, n=512) |
|---|---|---|---|
| Transformer (RoPE, RMSNorm, SwiGLU) | 3.6076 | 36.88 | 0.0044 |
| Mamba | 3.6454 | 38.30 | 0.0118 |

- **Blending** reaches its lowest measured loss (3.4924) at an interior
  mixing value ($\lambda=0.5$), below both pure baselines.
- **Latency crossover** measured at $n\approx1711$: the Transformer is
  faster below this sequence length, Mamba faster above it.
- **CKA similarity** between the two models' final hidden
  representations averages 0.8360 across held-out validation batches.

Full details, derivations, and discussion are in the dissertation paper.

## Repository Structure

```
├── data.py                    # WikiText-103 loading and tokenization
├── model_transformer.py       # Original 2017-style Transformer (reference/comparison)
├── model_transformer_v2.py    # Contemporary Transformer: RoPE, RMSNorm, SwiGLU (final M1)
├── model_mamba.py             # Mamba language model (official mamba_ssm reference impl.)
├── matching_config.py         # Parameter-matching search between M1 and M2
├── train.py                   # Training script for either baseline
├── measure_latency.py         # Single-length latency measurement (C1, C2)
├── latency_scaling.py         # Latency measured across a range of sequence lengths
├── hybrid_eval.py             # Stochastic routing: lambda sweep (L, C, J)
├── blend_eval.py              # Probabilistic blending: true blended-loss lambda sweep
├── cka.py                     # CKA representation similarity analysis
├── setup_env.sh               # One-time environment setup script
├── requirements.txt           # Frozen Python dependencies
├── lambda_sweep_results.csv   # Output of hybrid_eval.py
├── blend_sweep_results.csv    # Output of blend_eval.py
├── latency_scaling_results.csv# Output of latency_scaling.py
└── README.md
```

## Setup

All experiments were run on a single NVIDIA A40 GPU via QMUL's Data
Science Projects environment (JupyterHub).

```bash
chmod +x setup_env.sh
./setup_env.sh
```

This creates a virtual environment (`mamba_env`), installs PyTorch,
HuggingFace `transformers`/`datasets`, and the Mamba CUDA kernels
(`causal-conv1d`, `mamba-ssm`). The Mamba kernels require a real GPU to
install and run, so this step must be done on a GPU-enabled node.

## Running the Full Pipeline

### 1. Sanity checks

```bash
source mamba_env/bin/activate

python data.py                # confirms WikiText-103 loads and tokenizes
python model_transformer_v2.py# forward-pass + parameter count check (Transformer)
python model_mamba.py         # forward-pass + parameter count check (Mamba)
python matching_config.py     # searches for the Mamba n_layers that best matches
                               # the Transformer's parameter count
```

### 2. Train the baselines

```bash
python train.py --model transformer_v2 --epochs 3 --seq_len 512 --batch_size 16 \
                 --save_path ckpt_transformer_v2.pt

python train.py --model mamba --epochs 3 --seq_len 512 --batch_size 16 \
                 --save_path ckpt_mamba.pt
```

Note the final validation loss printed for each — these are $L_1$
(Transformer) and $L_2$ (Mamba).

For long runs, use `nohup ... &` so training survives if the browser tab
closes, e.g.:

```bash
nohup python train.py --model transformer_v2 --epochs 3 --seq_len 512 \
    --batch_size 16 --save_path ckpt_transformer_v2.pt \
    > log_transformer_v2.txt 2>&1 &
```

### 3. Measure inference latency ($C_1$, $C_2$)

```bash
python measure_latency.py --checkpoint ckpt_transformer_v2.pt --model transformer_v2
python measure_latency.py --checkpoint ckpt_mamba.pt --model mamba
```

### 4. Latency scaling across sequence lengths

Tests the $\mathcal{O}(n)$ vs $\mathcal{O}(n^2)$ claim properly, rather
than relying on a single measurement:

```bash
python latency_scaling.py --ckpt_transformer ckpt_transformer_v2.pt \
                           --ckpt_mamba ckpt_mamba.pt
```

Saves results to `latency_scaling_results.csv`.

### 5. Stochastic routing — lambda sweep

```bash
python hybrid_eval.py --L1 <transformer_val_loss> --L2 <mamba_val_loss> \
                       --C1 <transformer_latency> --C2 <mamba_latency> \
                       --beta 1.0
```

Computes $L(\lambda)$, $C(\lambda)$, $J(\lambda)$ across
$\lambda \in \{0, 0.25, 0.5, 0.75, 1\}$ and reports the selected
$\lambda^*$. Try several `--beta` values to see the sensitivity of the
selection to the efficiency weighting.

### 6. Probabilistic blending — true blend sweep

```bash
python blend_eval.py --ckpt_transformer ckpt_transformer_v2.pt \
                      --ckpt_mamba ckpt_mamba.pt \
                      --C1 <transformer_latency> --C2 <mamba_latency> \
                      --beta 1.0
```

Runs both models together on the validation set and computes the true
cross-entropy of the blended distribution $p(\lambda) = \lambda
\,\text{softmax}(z_1) + (1-\lambda)\,\text{softmax}(z_2)$ at each
$\lambda \in \{0, 0.1, \dots, 1.0\}$, rather than approximating it by
interpolating $L_1$ and $L_2$. Saves results to
`blend_sweep_results.csv`.

### 7. CKA representation analysis

```bash
python cka.py --ckpt_transformer ckpt_transformer_v2.pt --ckpt_mamba ckpt_mamba.pt
```

Reports the average CKA similarity between the two models' final hidden
representations across held-out validation batches.

## Notes

- All scripts assume a single GPU (`device="cuda"` by default).
- `model_transformer.py` (the original 2017-style architecture) is
  retained for reference and comparison; `model_transformer_v2.py`
  (RoPE, RMSNorm, SwiGLU) is the baseline actually used for the final
  reported results.
- Model sizes are kept modest (`d_model=256`, ~16M parameters) to fit
  the available compute budget. Increase if more compute is available,
  keeping the two architectures parameter-matched via
  `matching_config.py`.
- If you hit CUDA out-of-memory errors, reduce `--batch_size` or
  `--seq_len` first.

## Citation

If referencing this work, please cite the accompanying dissertation:

> Sheikh, R. (2026) *Architectural Mixing Strategies for Hybrid
> SSM-Attention Networks: A Systematic Study of Efficiency and
> Generalisation.* MSc Dissertation, Queen Mary University of London.