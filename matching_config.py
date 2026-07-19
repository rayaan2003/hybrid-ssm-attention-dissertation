"""
Utility to search for a Mamba configuration whose parameter count closely
matches a given Transformer configuration, so that comparisons between M1
and M2 are not confounded by differing model capacity.

Usage:
    python matching_config.py
"""

from model_transformer import TransformerLM, count_parameters as count_t
from model_mamba import MambaLM, count_parameters as count_m


def find_matching_mamba(vocab_size: int, transformer_kwargs: dict,
                         d_model: int, d_state: int = 16, d_conv: int = 4,
                         expand: int = 2, layer_search_range=range(2, 13)):
    """
    Builds the Transformer with `transformer_kwargs`, then searches over
    Mamba layer counts to find the closest parameter count match at the
    given d_model.
    """
    transformer = TransformerLM(vocab_size=vocab_size, **transformer_kwargs)
    target_params = count_t(transformer)
    print(f"Transformer parameter count: {target_params:,}")

    best_n_layers = None
    best_diff = float("inf")
    best_params = None

    for n_layers in layer_search_range:
        mamba = MambaLM(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv, expand=expand,
        )
        n_params = count_m(mamba)
        diff = abs(n_params - target_params)
        print(f"  Mamba n_layers={n_layers:2d} -> {n_params:,} params "
              f"(diff={diff:,})")
        if diff < best_diff:
            best_diff = diff
            best_n_layers = n_layers
            best_params = n_params

    print(f"\nBest match: n_layers={best_n_layers}, "
          f"params={best_params:,} (target={target_params:,}, "
          f"diff={best_diff:,}, "
          f"{100*best_diff/target_params:.2f}% off)")

    return best_n_layers, best_params, target_params


if __name__ == "__main__":
    # NOTE: constructing MambaLM instances requires a GPU (see model_mamba.py).
    # Run this on the HPC/JupyterHub GPU node.
    transformer_kwargs = dict(d_model=256, n_layers=4, n_heads=4, d_ff=1024)
    find_matching_mamba(
        vocab_size=50257,
        transformer_kwargs=transformer_kwargs,
        d_model=256,
    )
