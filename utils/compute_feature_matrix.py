from sympy import denom
import torch
import torch.nn as nn
from functools import partial
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from tqdm import tqdm


def add_colorbar(im, aspect=10, pad_fraction=0.2, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


class CrossModelCKA:
    """
    Computes the full layer-by-layer CKA heatmap between two models
    (e.g. original vs quantized): model_a[layer_i] vs model_b[layer_j]
    for ALL i, j pairs, producing an N x N matrix (same style as the
    standard CKA heatmap, just across two models instead of one).
    """

    def __init__(self,
                 model_a: nn.Module,
                 model_b: nn.Module,
                 layer_names: List[str],
                 device: str = 'cpu',
                 name_a: str = 'Original',
                 name_b: str = 'Quantized'):
        self.model_a = model_a.to(device).eval()
        self.model_b = model_b.to(device).eval()
        self.layer_names = layer_names
        self.device = device
        self.name_a = name_a
        self.name_b = name_b

        self.features_a = {}
        self.features_b = {}

        self._insert_hooks(self.model_a, self.features_a)
        self._insert_hooks(self.model_b, self.features_b)

        self.hsic_matrix = None

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
    def compare(self, dataloader, num_batches_limit=None):
        N = len(self.layer_names)
        self.hsic_matrix = torch.zeros(N, N, 3)

        total_batches = num_batches_limit if num_batches_limit else len(dataloader)

        for batch_idx, (x, *_) in enumerate(tqdm(dataloader, desc="| Comparing features |", total=total_batches)):
            if num_batches_limit and batch_idx >= num_batches_limit:
                break

            x = x.to(self.device)
            self.features_a.clear()
            self.features_b.clear()

            _ = self.model_a(x)
            _ = self.model_b(x)

            self._compare_cka(total_batches)

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (
            self.hsic_matrix[:, :, 0].sqrt() * self.hsic_matrix[:, :, 2].sqrt()
        )

        # denom = (self.hsic_matrix[:, :, 0].clamp(min=0).sqrt() * self.hsic_matrix[:, :, 2].clamp(min=0).sqrt() + 1e-8)
        # self.hsic_matrix = self.hsic_matrix[:, :, 1] / denom
        # # self.hsic_matrix = self.hsic_matrix[:, :, 1] / (
        # #     self.hsic_matrix[:, :, 0].sqrt() * self.hsic_matrix[:, :, 2].sqrt() + 1e-8
        # # )

    def _compare_cka(self, num_batches):
        for i, name_i in enumerate(self.layer_names):
            X = self._flatten(self.features_a[name_i])
            K = X @ X.t()
            K.fill_diagonal_(0.0)
            self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

            for j, name_j in enumerate(self.layer_names):
                Y = self._flatten(self.features_b[name_j])
                L = Y @ Y.t()
                L.fill_diagonal_(0.0)

                assert K.shape == L.shape, f"Feature shape mismatch! {K.shape}, {L.shape}"

                self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

    def plot_cka(self, save_path: str = None, title: str = None, show: bool = True):
        if self.hsic_matrix is None:
            raise RuntimeError("Call .compare(dataloader) before .plot_cka().")

        fig, ax = plt.subplots(figsize=(6, 5.25))
        im = ax.imshow(self.hsic_matrix.cpu(), origin='lower', cmap='magma')

        ax.set_xlabel(f"{self.name_b} Layers", fontsize=16)
        ax.set_ylabel(f"{self.name_a} Layers", fontsize=16)

        labels = range(self.hsic_matrix.shape[0])
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        ax.set_title(title or f"CKA: {self.name_a} vs {self.name_b}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if show:
            plt.show()

        return fig

def compute_cka_similarity(
        model: nn.Module,
        layer_names: List[str],
        test_loader: torch.utils.data.DataLoader,
        device: str = 'cpu',
        batch_limit=None,
):
    """
    Computes the layer-by-layer CKA matrix within a single model:
    """
    model = model.to(device).eval()

    features = {}

    def _log_layer(name, layer, inp, out):
        features[name] = out.detach()

    for name, layer in model.named_modules():
        if name in layer_names:
            layer.register_forward_hook(partial(_log_layer, name))

    def _HSIC(K, L):
        N = K.shape[0]
        ones = torch.ones(N, 1, device=device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def _flatten(feat):
        return feat.reshape(feat.shape[0], -1)

    N = len(layer_names)
    hsic_matrix = torch.zeros(N, N, 3)

    total_batches = batch_limit if batch_limit else len(test_loader)

    with torch.no_grad():
        for batch_idx, (x, *_) in enumerate(tqdm(test_loader, desc="| Comparing features |", total=total_batches)):
            if batch_limit and batch_idx >= batch_limit:
                break

            x = x.to(device)
            features.clear()

            _ = model(x)

            for i, name_i in enumerate(layer_names):
                X = _flatten(features[name_i])
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                hsic_matrix[i, :, 0] += _HSIC(K, K) / total_batches

                for j, name_j in enumerate(layer_names):
                    Y = _flatten(features[name_j])
                    L = Y @ Y.t()
                    L.fill_diagonal_(0.0)

                    assert K.shape == L.shape, f"Feature shape mismatch! {K.shape}, {L.shape}"

                    hsic_matrix[i, j, 1] += _HSIC(K, L) / total_batches
                    hsic_matrix[i, j, 2] += _HSIC(L, L) / total_batches

    cka_matrix = hsic_matrix[:, :, 1] / (hsic_matrix[:, :, 0].sqrt() * hsic_matrix[:, :, 2].sqrt())

    return cka_matrix

def plot_cka_matrix(cka_matrix, save_path=None, title=None, show=True):
    fig, ax = plt.subplots(figsize=(6, 5.25))
    im = ax.imshow(cka_matrix.cpu(), origin='lower', cmap='magma')

    ax.set_xlabel("Layers", fontsize=16)
    ax.set_ylabel("Layers", fontsize=16)

    labels = range(cka_matrix.shape[0])
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_title(title or "CKA Similarity", fontsize=18)

    add_colorbar(im)
    plt.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()

    return fig

def compare_cka_and_print_result(
        args,
        model: nn.Module,
        quantized_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        batch_limit=None,
        save_path='cka_comparison.png',
        title=None,
        name_a='Original',
        name_b='Quantized'
):
    if args.model == 'mvit':
        layer_names = ['qkv', 'proj', 'fc1', 'fc2']
    elif args.model == 'mf':
        layer_names = ['to_q', 'to_kv', 'to_memory', 'to_out', 'nn1', 'nn2']
    else: # add them if necessary
        layer_names = []

    if isinstance(layer_names, str):
        layer_names = [layer_names]

    matched_layer_names = []
    for name, module in model.named_modules():
        if any(ln in name for ln in layer_names) and isinstance(module, torch.nn.Linear):
            matched_layer_names.append(name)
    
    print(f"Matched {len(matched_layer_names)} layers:")
    for layer in matched_layer_names:
        print(f"  - {layer}")

    # first do cka for models one by one 
    cka_original = compute_cka_similarity(model, matched_layer_names, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu', batch_limit=batch_limit)
    plot_cka_matrix(cka_original, save_path=f'cka_original_{args.model}_nbits{args.nbits}.png', title=f'CKA for {args.model} original')

    cka_quantized = compute_cka_similarity(quantized_model, matched_layer_names, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu', batch_limit=batch_limit)
    plot_cka_matrix(cka_quantized, save_path=f'cka_quantized_{args.model}_nbits{args.nbits}.png', title=f'CKA for {args.model} quantized')

    #cka both models together
    cka = CrossModelCKA(
        model_a=model,
        model_b=quantized_model,
        layer_names=matched_layer_names,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        name_a=name_a,
        name_b=name_b
    )
    cka.compare(test_loader, num_batches_limit=batch_limit)

    cka.plot_cka(
        save_path=save_path,
        title=title or f'CKA: {name_a} vs {name_b} for layers: {layer_names}'
    )

    return cka