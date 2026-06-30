import torch
import torch.nn as nn
from functools import partial
from typing import List, Dict


class CrossModelCKA:
    """
    Computes per-layer CKA between two models (e.g. original vs quantized)
    compares model_a[layer_i] against model_b[layer_i] instead of
    layer_i vs layer_j within one model.
    """
    def __init__(self,
                 model_a: nn.Module,
                 model_b: nn.Module,
                 layer_names: List[str],
                 device: str = 'cpu'):
        self.model_a = model_a.to(device).eval()
        self.model_b = model_b.to(device).eval()
        self.layer_names = layer_names
        self.device = device

        self.features_a = {}
        self.features_b = {}

        self._insert_hooks(self.model_a, self.features_a)
        self._insert_hooks(self.model_b, self.features_b)

    def _log_layer(self, store, name, layer, inp, out):
        store[name] = out.detach()

    def _insert_hooks(self, model, store):
        for name, layer in model.named_modules():
            if name in self.layer_names:
                layer.register_forward_hook(partial(self._log_layer, store, name))

    def _HSIC(self, K, L):
        N = K.shape[0]
        ones = torch.ones(N, 1, device=self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def _flatten(self, feat):
        return feat.reshape(feat.shape[0], -1)

    @torch.no_grad()
    def compare(self, dataloader, num_batches_limit=None) -> Dict[str, float]:
        cka_sums = {name: 0.0 for name in self.layer_names}
        n_batches = 0

        for batch_idx, (x, *_) in enumerate(dataloader):
            if num_batches_limit and batch_idx >= num_batches_limit:
                break

            x = x.to(self.device)
            self.features_a.clear()
            self.features_b.clear()

            _ = self.model_a(x)
            _ = self.model_b(x)

            for name in self.layer_names:
                if name not in self.features_a or name not in self.features_b:
                    continue

                X = self._flatten(self.features_a[name])
                Y = self._flatten(self.features_b[name])

                K = X @ X.t()
                K.fill_diagonal_(0.0)
                L = Y @ Y.t()
                L.fill_diagonal_(0.0)

                hsic_kl = self._HSIC(K, L)
                hsic_kk = self._HSIC(K, K)
                hsic_ll = self._HSIC(L, L)

                cka = hsic_kl / ((hsic_kk ** 0.5) * (hsic_ll ** 0.5) + 1e-8)
                cka_sums[name] += cka

            n_batches += 1

        return {name: cka_sums[name] / n_batches for name in self.layer_names}
    

import matplotlib.pyplot as plt

def plot_cka_bar(cka_scores: dict, save_path: str = None, title: str = None):
    layers = list(cka_scores.keys())
    values = list(cka_scores.values())

    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.8), 5))
    bars = ax.bar(range(len(layers)), values, color='#1D9E75')

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('CKA similarity')
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                ha='center', fontsize=8)

    ax.set_title(title or 'CKA similarity: original vs quantized', fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def compare_cka_and_print_result(
        layer_name,
        model: nn.Module,
        quantized_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        batch_limit = None,
        save_path = 'cka_comparison.png'
):
    matched_layer_names = []
    for name, module in model.named_modules():
        if layer_name in name and isinstance(module, torch.nn.Linear):
            matched_layer_names.append(name)

    print(matched_layer_names)

    cka = CrossModelCKA(
        model_a = model,
        model_b = quantized_model,
        layer_names = matched_layer_names,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    scores = cka.compare(test_loader, num_batches_limit=batch_limit)

    plot_cka_bar(scores, save_path=save_path, title='CKA similarity: original vs quantized for layer: "{}"'.format(layer_name))