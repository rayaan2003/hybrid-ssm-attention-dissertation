# Hybrid SSM-Attention Dissertation Code

Implements the methodology described in the dissertation: a Transformer
baseline (M1), a Mamba baseline (M2), a stochastic routing hybrid
controlled by lambda, and CKA representation analysis between the two.

## 1. Setup (run once, on the JupyterHub GPU terminal)

```bash
chmod +x setup_env.sh
./setup_env.sh
```

This creates a virtual environment `mamba_env`, installs PyTorch,
HuggingFace `transformers`/`datasets`, and the Mamba CUDA kernels
(`causal-conv1d`, `mamba-ssm`). The Mamba kernels require a real GPU to
install and run, so this step must be done on the GPU node, not locally.

## 2. Sanity checks

```bash
source ~/mamba_env/bin/activate
python data.py                # quick check that WikiText-103 loads and tokenizes
python model_transformer.py   # quick forward-pass + parameter count check
python model_mamba.py         # same, for Mamba (needs GPU)
python matching_config.py     # searches for a Mamba n_layers that matches
                               # the Transformer's parameter count
```

Update the `n_layers` used in `train.py`, `measure_latency.py`, and
`cka.py` (currently set to 6) based on whatever `matching_config.py`
finds to be the closest match.

## 3. Train the two baselines

```bash
python train.py --model transformer --epochs 3 --save_path ckpt_transformer.pt
python train.py --model mamba       --epochs 3 --save_path ckpt_mamba.pt
```

Each run prints validation loss and perplexity at the end of every epoch.
Note down the final validation loss for each model -- these are your L1
(Transformer) and L2 (Mamba) values.

Adjust `--epochs`, `--batch_size`, and `--seq_len` depending on how much
GPU time/memory you have available. Start small (e.g. `--epochs 1
--seq_len 256`) to confirm everything runs end-to-end before committing
to a longer run.

## 4. Measure inference latency (C1, C2)

```bash
python measure_latency.py --checkpoint ckpt_transformer.pt --model transformer
python measure_latency.py --checkpoint ckpt_mamba.pt --model mamba
```

Note down the reported latency-per-token for each -- these are your C1
and C2 values.

## 5. Run the lambda sweep

```bash
python hybrid_eval.py --L1 <transformer_val_loss> --L2 <mamba_val_loss> \
                       --C1 <transformer_latency> --C2 <mamba_latency> \
                       --beta 1.0
```

This prints a table of L(lambda), C(lambda), J(lambda) for
lambda = 0, 0.25, 0.5, 0.75, 1, saves it to `lambda_sweep_results.csv`,
and reports the selected lambda* (the value minimising J). This directly
produces the numbers for the Results section of the dissertation.

Try a couple of different `--beta` values to see how sensitive lambda* is
to how much you weight efficiency vs. accuracy -- worth reporting both in
the write-up.

## 6. CKA representation analysis

```bash
python cka.py --ckpt_transformer ckpt_transformer.pt --ckpt_mamba ckpt_mamba.pt
```

Reports the average CKA similarity score between the Transformer's and
Mamba's final hidden representations, across several validation batches.
This is the number that goes into the Representation Analysis section.

## Notes

- All scripts assume a single GPU (`device="cuda"` by default).
- Model sizes in `train.py`, `measure_latency.py`, and `cka.py` are
  currently small (`d_model=256`) to keep training times reasonable on a
  single GPU. Increase if you have more compute time available, but keep
  the Transformer and Mamba configs matched via `matching_config.py`.
- If you hit CUDA out-of-memory errors, reduce `--batch_size` or
  `--seq_len` first.
